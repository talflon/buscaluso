#[cfg(test)]
mod tests;

use std::borrow::Borrow;
use std::cmp::min;
use std::collections::BTreeMap;
use std::fmt::Debug;

use unicode_normalization::UnicodeNormalization;

use crate::FonError::*;
use crate::{Fon, FonRegistry, Result, NO_FON_CHAR};

pub trait Normalizer {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<Fon>) -> Result<()>;

    fn normalize(&self, word: &str) -> Result<Vec<Fon>> {
        let mut normalized = Vec::new();
        self.normalize_into(word, &mut normalized)?;
        Ok(normalized)
    }
}

pub fn prenormalized_chars(word: &str) -> impl Iterator<Item = char> + '_ {
    word.nfc().flat_map(char::to_lowercase)
}

impl Normalizer for FonRegistry {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<Fon>) -> Result<()> {
        for c in prenormalized_chars(word) {
            normalized.push(self.get_fon(c)?);
        }
        Ok(())
    }
}

type NormalizeRuleMap = BTreeMap<Box<[char]>, Box<[Fon]>>;

#[derive(Debug, Clone)]
pub struct NormalizeRuleSet {
    /// Stores rules per length of what to replace.
    /// As no rules can replace an empty string, `rules[0]` is for length 1.
    rules: Vec<NormalizeRuleMap>,
}

impl NormalizeRuleSet {
    pub fn new() -> NormalizeRuleSet {
        NormalizeRuleSet {
            rules: vec![BTreeMap::from([(
                Box::from(&[NO_FON_CHAR] as &[char]),
                Box::from(&[] as &[Fon]),
            )])],
        }
    }

    fn rule_len_idx(len: usize) -> usize {
        len - 1
    }

    pub fn longest_rule(&self) -> usize {
        self.rules.len()
    }

    pub fn add_rule(&mut self, pattern: &[char], normed: &[Fon]) -> Result<()> {
        while self.longest_rule() < pattern.len() {
            self.rules.push(BTreeMap::new());
        }
        let m = &mut self.rules[Self::rule_len_idx(pattern.len())];
        if !m.contains_key(pattern) {
            m.insert(pattern.into(), normed.into());
            Ok(())
        } else {
            Err(DuplicateNormRule(pattern.into()))
        }
    }

    pub fn get_rule(&self, pattern: &[char]) -> Option<&[Fon]> {
        self.rules
            .get(Self::rule_len_idx(pattern.len()))
            .and_then(|m| m.get(pattern).map(Box::borrow))
    }

    pub fn find_rule(&self, input: &[char]) -> Option<(usize, &[Fon])> {
        for len in (1..=min(input.len(), self.longest_rule())).rev() {
            if let Some(rule) = self.get_rule(&input[..len]) {
                return Some((len, rule));
            }
        }
        None
    }
}

pub trait RuleBasedNormalizer {
    fn rule_set(&self) -> &NormalizeRuleSet;
    fn fon_registry(&self) -> &FonRegistry;
}

impl<N: RuleBasedNormalizer> Normalizer for N {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<Fon>) -> Result<()> {
        if word.contains(NO_FON_CHAR) {
            return Err(NoSuchFon(NO_FON_CHAR));
        }
        // surround input with NO_FON_CHAR to handle anchored normalization rules
        let mut input: Vec<char> = vec![NO_FON_CHAR];
        input.extend(prenormalized_chars(word));
        input.push(NO_FON_CHAR);
        // treat input as a mutable slice so we can easily drop from the front by reassigning
        let mut input: &[char] = &input;
        while !input.is_empty() {
            if let Some((len, result)) = self.rule_set().find_rule(input) {
                normalized.extend_from_slice(result);
                input = &input[len..];
            } else {
                // fall back to registry, to output a single char.
                normalized.push(self.fon_registry().get_fon(input[0])?);
                input = &input[1..];
            }
        }
        // we're done because our result has been pushed into `normalized`
        Ok(())
    }
}

// allows for easier testing of trait
impl<'a, 'b> RuleBasedNormalizer for (&'a NormalizeRuleSet, &'b FonRegistry) {
    fn rule_set(&self) -> &NormalizeRuleSet {
        self.0
    }

    fn fon_registry(&self) -> &FonRegistry {
        self.1
    }
}
