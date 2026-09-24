//! Native matcher for pii_rules.
//!
//! It holds no rule of its own. pii_rules.py hands over its pattern strings,
//! their required keywords, and the characters it strips, so the rules keep one
//! source of truth. This module only finds the matches, faster than `re` does:
//! PCRE2 compiles each pattern to machine code, and one Aho-Corasick pass over
//! the text decides which patterns can match at all.
//!
//! PCRE2 and Python's `re` agree on every construct the rules use, except on
//! the characters their Unicode tables disagree on. pii_rules derives those at
//! load with class_members, a text that holds one is declined, and pii_rules
//! scans it with `re` instead.

use std::collections::HashSet;

use aho_corasick::AhoCorasick;
use pcre2::bytes::{CaptureLocations, Regex, RegexBuilder};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// pii_rules refuses a module whose version differs from the one it expects,
/// so a stale build cannot answer for rules it was not built against.
const ENGINE_VERSION: u32 = 2;

/// A pattern as pii_rules hands it over: the source with PCRE2's own \w and
/// \b, the source with them spelled the way `re` reads them, whether it
/// ignores case, the named groups it reports, and the keywords it cannot match
/// without. An empty keyword list means the pattern always runs.
type PatternSource = (String, String, bool, Vec<String>, Vec<String>);

/// A match in code point offsets: the whole match, then each requested group.
type Found = (usize, usize, Vec<Option<(usize, usize)>>);

/// The two forms agree on a text without the characters `word_differs`
/// finds. The native form keeps PCRE2's start-of-match optimizations, which a
/// \b spelled as lookarounds loses, and costs half as much.
struct Pattern {
    native: Regex,
    spelled: Regex,
    group_slots: Vec<Option<usize>>,
    keyword_ids: Vec<usize>,
}

#[pyclass(frozen)]
struct Engine {
    patterns: Vec<Pattern>,
    keywords: AhoCorasick,
    keyword_count: usize,
    decline: HashSet<char>,
    word_differs: Regex,
    invisible: HashSet<char>,
}

#[pymethods]
impl Engine {
    #[new]
    fn new(
        sources: Vec<PatternSource>,
        decline: &str,
        word_differs: &str,
        invisible: &str,
    ) -> PyResult<Self> {
        let mut patterns = Vec::with_capacity(sources.len());
        let mut keyword_list: Vec<String> = Vec::new();
        for (native_source, spelled_source, ignore_case, groups, keywords) in sources {
            let native = compile(&native_source, ignore_case)?;
            let spelled = compile(&spelled_source, ignore_case)?;
            let names = spelled.capture_names().to_vec();
            let group_slots = groups
                .iter()
                .map(|group| {
                    names
                        .iter()
                        .position(|name| name.as_deref() == Some(group.as_str()))
                })
                .collect();
            let first = keyword_list.len();
            keyword_list.extend(keywords);
            patterns.push(Pattern {
                native,
                spelled,
                group_slots,
                keyword_ids: (first..keyword_list.len()).collect(),
            });
        }
        // ASCII case folding only. pii_rules declines every text in which a
        // non-ASCII character would fold onto an ASCII keyword, so this cannot
        // skip a pattern that `re` would have matched.
        let keywords = AhoCorasick::builder()
            .ascii_case_insensitive(true)
            .build(&keyword_list)
            .map_err(|error| PyValueError::new_err(format!("keywords: {error}")))?;
        Ok(Engine {
            patterns,
            keywords,
            keyword_count: keyword_list.len(),
            decline: decline.chars().collect(),
            word_differs: compile(word_differs, false)?,
            invisible: invisible.chars().collect(),
        })
    }

    /// Return the matches of each pattern, in the order they were handed over,
    /// or None when the text holds a character this engine does not match
    /// exactly as `re` does.
    fn scan(&self, text: &str) -> PyResult<Option<Vec<Vec<Found>>>> {
        let bytes = text.as_bytes();
        let ascii = text.is_ascii();
        if !ascii
            && text
                .chars()
                .any(|character| self.decline.contains(&character))
        {
            return Ok(None);
        }
        let mut present = vec![false; self.keyword_count];
        for hit in self.keywords.find_overlapping_iter(bytes) {
            present[hit.pattern().as_usize()] = true;
        }
        let spelled = !ascii && self.word_differs.is_match(bytes).map_err(match_error)?;
        let offsets = CharOffsets::new(text, ascii);
        let mut found = Vec::with_capacity(self.patterns.len());
        for pattern in &self.patterns {
            let gated = !pattern.keyword_ids.is_empty()
                && !pattern.keyword_ids.iter().any(|id| present[*id]);
            if gated {
                found.push(Vec::new());
                continue;
            }
            let regex = if spelled {
                &pattern.spelled
            } else {
                &pattern.native
            };
            found.push(find_all(regex, &pattern.group_slots, text, &offsets)?);
        }
        Ok(Some(found))
    }

    /// The same contract as pii_rules._strip_invisibles: None when the text is
    /// clean, else the clean text and the original index of each character,
    /// plus one final entry holding the length of the text.
    fn strip_invisibles(&self, text: &str) -> Option<(String, Vec<usize>)> {
        if !text
            .chars()
            .any(|character| self.invisible.contains(&character))
        {
            return None;
        }
        let mut clean = String::with_capacity(text.len());
        let mut index_map = Vec::with_capacity(text.len() + 1);
        let mut length = 0;
        for (index, character) in text.chars().enumerate() {
            length = index + 1;
            if self.invisible.contains(&character) {
                continue;
            }
            clean.push(character);
            index_map.push(index);
        }
        index_map.push(length);
        Some((clean, index_map))
    }
}

fn compile(source: &str, ignore_case: bool) -> PyResult<Regex> {
    RegexBuilder::new()
        .utf(true)
        .ucp(true)
        .jit_if_available(true)
        .caseless(ignore_case)
        .build(source)
        .map_err(|error| PyValueError::new_err(format!("{source}: {error}")))
}

fn match_error(error: pcre2::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Every non-overlapping match, left to right, the way `re.finditer` walks.
fn find_all(
    regex: &Regex,
    group_slots: &[Option<usize>],
    text: &str,
    offsets: &CharOffsets,
) -> PyResult<Vec<Found>> {
    let bytes = text.as_bytes();
    let mut locations: CaptureLocations = regex.capture_locations();
    let mut matches = Vec::new();
    let mut start = 0;
    while start <= bytes.len() {
        let matched = regex
            .captures_read_at(&mut locations, bytes, start)
            .map_err(match_error)?;
        let Some(whole) = matched else { break };
        let groups = group_slots
            .iter()
            .map(|slot| {
                slot.and_then(|slot| locations.get(slot))
                    .map(|(group_start, group_end)| {
                        (offsets.at(group_start), offsets.at(group_end))
                    })
            })
            .collect();
        matches.push((offsets.at(whole.start()), offsets.at(whole.end()), groups));
        start = next_start(text, whole.start(), whole.end());
    }
    Ok(matches)
}

/// An empty match must still move the search on, by one whole character.
fn next_start(text: &str, match_start: usize, match_end: usize) -> usize {
    if match_end > match_start {
        return match_end;
    }
    let mut next = match_end + 1;
    while next < text.len() && !text.is_char_boundary(next) {
        next += 1;
    }
    next
}

/// Maps a UTF-8 byte offset to the code point offset Python indexes by.
struct CharOffsets {
    table: Option<Vec<usize>>,
}

impl CharOffsets {
    fn new(text: &str, ascii: bool) -> Self {
        if ascii {
            return CharOffsets { table: None };
        }
        let mut table = vec![0; text.len() + 1];
        let mut count = 0;
        for (byte_offset, _) in text.char_indices() {
            table[byte_offset] = count;
            count += 1;
        }
        table[text.len()] = count;
        CharOffsets { table: Some(table) }
    }

    fn at(&self, byte_offset: usize) -> usize {
        match &self.table {
            Some(table) => table[byte_offset],
            None => byte_offset,
        }
    }
}

/// The Unicode version of PCRE2's character tables. When it equals Python's,
/// pii_rules can skip deriving the characters the two tables disagree on.
fn unicode_version() -> String {
    let mut buffer = [0u8; 32];
    // SAFETY: the buffer is larger than the longest version string PCRE2
    // writes, and the call writes at most that string and its terminator.
    let written = unsafe {
        pcre2_sys::pcre2_config_8(
            pcre2_sys::PCRE2_CONFIG_UNICODE_VERSION,
            buffer.as_mut_ptr().cast(),
        )
    };
    match usize::try_from(written) {
        Ok(length) if length > 0 => String::from_utf8_lossy(&buffer[..length - 1]).into_owned(),
        _ => String::new(),
    }
}

/// The characters of text that a one-character pattern matches, in order.
/// pii_rules runs the same class through re over every code point, and the
/// characters on which the two answers differ are the ones Engine declines.
#[pyfunction]
fn class_members(pattern: &str, ignore_case: bool, text: &str) -> PyResult<String> {
    let regex = compile(pattern, ignore_case)?;
    let mut members = String::new();
    for found in regex.find_iter(text.as_bytes()) {
        let found = found.map_err(match_error)?;
        members.push_str(&text[found.start()..found.end()]);
    }
    Ok(members)
}

#[pymodule]
fn pii_rules_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ENGINE_VERSION", ENGINE_VERSION)?;
    module.add("UNICODE_VERSION", unicode_version())?;
    module.add_class::<Engine>()?;
    module.add_function(wrap_pyfunction!(class_members, module)?)
}
