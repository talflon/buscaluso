use std::borrow::Borrow;
use std::cmp::min;
use std::collections::BTreeMap;
use std::io::BufRead;
use std::{io, iter};

use nom::Finish;
use rulefile::Rule;
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

mod rulefile;

#[cfg(test)]
mod tests;

pub type Result<T> = std::result::Result<T, FonError>;

#[derive(Error, Debug)]
pub enum FonError {
    #[error("No such fon {0:?}")]
    NoSuchFon(char),

    #[error("Ran out of fon ids")]
    NoMoreFonIds,

    #[error("No such fon id {0}")]
    NoSuchFonId(FonId),

    #[error("Already had a normalization rule for {0:?}")]
    DuplicateNormRule(Box<[char]>),

    #[error("Invalid normalization rule: {0}")]
    InvalidNormRule(String),

    #[error("No such alias {0:?}")]
    NoSuchAlias(String),

    #[error("IO error {source:?}")]
    Io {
        #[from]
        source: io::Error,
    },

    #[error("Parsing error on line {line_no}: {text:?}")]
    ParseErr { line_no: usize, text: String },
}

use FonError::*;

const MAX_FON_ID: FonId = 127;

type FonId = u8;

pub const NO_FON: FonId = 0;

pub const NO_FON_CHAR: char = '_';

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct FonSet {
    bits: u128,
}

impl FonSet {
    pub const EMPTY: FonSet = FonSet { bits: 0 };

    pub fn new() -> FonSet {
        FonSet::EMPTY
    }

    pub fn contains(&self, id: FonId) -> bool {
        self.bits & (1 << id) != 0
    }

    pub fn len(&self) -> usize {
        self.bits.count_ones() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub fn is_real(&self) -> bool {
        !self.contains(NO_FON)
    }

    pub fn iter(&self) -> FonSetIter {
        FonSetIter {
            fonset: self.clone(),
            index: 0,
        }
    }

    pub fn fons(&self, reg: &FonRegistry) -> Result<Vec<char>> {
        let mut result: Vec<char> = Vec::new();
        for i in self.iter() {
            result.push(reg.get_fon(i)?);
        }
        Ok(result)
    }

    pub fn seq_is_empty<S: AsRef<[FonSet]>>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.is_empty() || slice.iter().any(|s| s.is_empty())
    }

    pub fn seq_is_real<S: AsRef<[FonSet]>>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.is_empty() || slice.iter().all(|s| s.is_real())
    }

    pub fn seq_is_valid<S: AsRef<[FonSet]>>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.len() <= 1 || slice[1..slice.len() - 1].iter().all(|s| s.is_real())
    }

    pub fn seq_match_at_into<P: AsRef<[FonSet]>, S: AsRef<[FonSet]>>(
        pattern: &P,
        seq: &S,
        index: usize,
        result: &mut Vec<FonSet>,
    ) -> bool {
        let pattern_slice = pattern.as_ref();
        let seq_slice = seq.as_ref();
        if index + pattern_slice.len() > seq_slice.len() {
            false
        } else {
            result.clear();
            result.extend_from_slice(seq_slice);
            for i in 0..pattern_slice.len() {
                result[index + i] &= pattern_slice[i];
            }
            !FonSet::seq_is_empty(result)
        }
    }

    pub fn seq_match_at<P: AsRef<[FonSet]>, S: AsRef<[FonSet]>>(
        pattern: &P,
        seq: &S,
        index: usize,
    ) -> Option<Vec<FonSet>> {
        let mut result = Vec::new();
        if Self::seq_match_at_into(pattern, seq, index, &mut result) {
            Some(result)
        } else {
            None
        }
    }
}

impl From<FonId> for FonSet {
    fn from(id: FonId) -> Self {
        FonSet { bits: 1 << id }
    }
}

impl<const N: usize> From<[FonId; N]> for FonSet {
    fn from(fons: [FonId; N]) -> Self {
        let mut s = FonSet::new();
        for f in fons {
            s |= f;
        }
        s
    }
}

impl std::ops::BitOr for FonSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits | rhs.bits,
        }
    }
}

impl std::ops::BitOr<FonId> for FonSet {
    type Output = Self;

    fn bitor(self, rhs: FonId) -> Self::Output {
        self | FonSet::from(rhs)
    }
}

impl std::ops::BitOrAssign for FonSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

impl std::ops::BitOrAssign<FonId> for FonSet {
    fn bitor_assign(&mut self, rhs: FonId) {
        *self |= FonSet::from(rhs)
    }
}

impl std::ops::BitAnd for FonSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits & rhs.bits,
        }
    }
}

impl std::ops::BitAndAssign for FonSet {
    fn bitand_assign(&mut self, rhs: Self) {
        self.bits &= rhs.bits;
    }
}

impl std::ops::Sub for FonSet {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            bits: self.bits & !rhs.bits,
        }
    }
}

impl std::ops::Sub<FonId> for FonSet {
    type Output = Self;

    fn sub(self, rhs: FonId) -> Self::Output {
        self - FonSet::from(rhs)
    }
}

impl std::ops::SubAssign for FonSet {
    fn sub_assign(&mut self, rhs: Self) {
        self.bits &= !rhs.bits;
    }
}

impl std::ops::SubAssign<FonId> for FonSet {
    fn sub_assign(&mut self, rhs: FonId) {
        *self -= FonSet::from(rhs)
    }
}

pub struct FonSetIter {
    fonset: FonSet,
    index: usize,
}

impl Iterator for FonSetIter {
    type Item = FonId;
    fn next(&mut self) -> Option<FonId> {
        if self.fonset.is_empty() {
            None
        } else {
            // skip all zeros, get nonzero index, then move one past
            let zeros = self.fonset.bits.trailing_zeros();
            self.index += zeros as usize;
            let item = Some(self.index as FonId);
            self.fonset.bits >>= zeros;
            self.fonset.bits >>= 1;
            self.index += 1;
            item
        }
    }
}

impl IntoIterator for FonSet {
    type Item = FonId;
    type IntoIter = FonSetIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<FonId> for FonSet {
    fn from_iter<I: IntoIterator<Item = FonId>>(iter: I) -> Self {
        let mut s = FonSet::new();
        for i in iter {
            s |= i;
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct FonRegistry {
    fons: Vec<char>,
}

impl FonRegistry {
    pub fn new() -> FonRegistry {
        FonRegistry {
            fons: vec![NO_FON_CHAR],
        }
    }

    pub fn add(&mut self, fon: char) -> Result<FonId> {
        match self.try_get_id(fon) {
            Some(id) => Ok(id),
            None => {
                let id = self.fons.len() as FonId;
                if id <= MAX_FON_ID {
                    self.fons.push(fon);
                    Ok(id)
                } else {
                    Err(NoMoreFonIds)
                }
            }
        }
    }

    pub fn try_get_id(&self, fon: char) -> Option<FonId> {
        self.fons.iter().position(|&f| f == fon).map(|i| i as FonId)
    }

    pub fn get_id(&self, fon: char) -> Result<FonId> {
        self.try_get_id(fon).ok_or(NoSuchFon(fon))
    }

    pub fn try_get_fon(&self, id: FonId) -> Option<char> {
        self.fons.get(id as usize).cloned()
    }

    pub fn get_fon(&self, id: FonId) -> Result<char> {
        self.try_get_fon(id).ok_or(NoSuchFonId(id))
    }
}

trait Normalizer {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<FonId>) -> Result<()>;

    fn normalize(&self, word: &str) -> Result<Vec<FonId>> {
        let mut normalized = Vec::new();
        self.normalize_into(word, &mut normalized)?;
        Ok(normalized)
    }
}

fn prenormalized_chars(word: &str) -> impl Iterator<Item = char> + '_ {
    word.nfc().flat_map(char::to_lowercase)
}

impl Normalizer for FonRegistry {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<FonId>) -> Result<()> {
        for c in prenormalized_chars(word) {
            normalized.push(self.get_id(c)?);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NormalizeRuleSet {
    rules: Vec<BTreeMap<Box<[char]>, Box<[FonId]>>>,
}

impl NormalizeRuleSet {
    fn new() -> NormalizeRuleSet {
        NormalizeRuleSet { rules: Vec::new() }
    }

    fn rule_len_idx(len: usize) -> usize {
        len - 1
    }

    pub fn longest_rule(&self) -> usize {
        self.rules.len()
    }

    pub fn add_rule(&mut self, pattern: &[char], normed: &[FonId]) -> Result<()> {
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

    pub fn get_rule(&self, pattern: &[char]) -> Option<&[FonId]> {
        self.rules
            .get(Self::rule_len_idx(pattern.len()))
            .and_then(|m| m.get(pattern).map(Box::borrow))
    }

    pub fn find_rule(&self, input: &[char]) -> Option<(usize, &[FonId])> {
        for len in (1..=min(input.len(), self.longest_rule())).rev() {
            if let Some(rule) = self.get_rule(&input[..len]) {
                return Some((len, rule));
            }
        }
        None
    }
}

trait RuleBasedNormalizer {
    fn rule_set(&self) -> &NormalizeRuleSet;
    fn fon_registry(&self) -> &FonRegistry;
}

impl<N: RuleBasedNormalizer> Normalizer for N {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<FonId>) -> Result<()> {
        // surround input with NO_FON_CHAR to handle anchored normalization rules
        let mut input: Vec<char> = Vec::new();
        input.push(NO_FON_CHAR);
        input.extend(prenormalized_chars(word));
        input.push(NO_FON_CHAR);
        // treat input as a mutable slice so we can easily drop from the front by reassigning
        let mut input: &[char] = &input;
        while !input.is_empty() {
            if let Some((len, result)) = self.rule_set().find_rule(input) {
                normalized.extend_from_slice(result);
                input = &input[len..];
            } else {
                // fall back to registry, to output a single char. but ignore end anchors
                if input[0] != NO_FON_CHAR {
                    normalized.push(self.fon_registry().get_id(input[0])?);
                }
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

pub type Cost = i32;

#[derive(Debug, Clone)]
pub struct BuscaCfg {
    fon_registry: FonRegistry,
    dictionary: BTreeMap<Box<[FonId]>, Vec<Box<str>>>,
    normalize_rules: NormalizeRuleSet,
    aliases: BTreeMap<String, FonSet>,
}

impl BuscaCfg {
    pub fn new() -> BuscaCfg {
        BuscaCfg {
            fon_registry: FonRegistry::new(),
            dictionary: BTreeMap::new(),
            normalize_rules: NormalizeRuleSet::new(),
            aliases: BTreeMap::new(),
        }
    }

    pub fn load_rules<R: BufRead>(&mut self, input: R) -> Result<()> {
        for (line_no, line) in input.lines().enumerate() {
            match rulefile::rule_line(&line?).finish() {
                Ok((_, Some(rule))) => self.add_rule(rule),
                Ok((_, None)) => Ok(()),
                Err(x) => Err(ParseErr {
                    line_no: line_no + 1,
                    text: x.input.to_owned(),
                }),
            }?;
        }
        Ok(())
    }

    fn add_rule(&mut self, rule: Rule) -> Result<()> {
        match rule {
            Rule::Alias(name, items) => {
                let fons = self.resolve_rule_item_set(&items)?;
                self.set_alias(name.into(), fons);
            }
            Rule::Norm { from, to } => {
                let from_charsets = self.resolve_norm_rule_lhs(&from)?;
                let to_fons = self.resolve_norm_rule_result(&to)?;
                for_cartesian_product(&from_charsets, |from_chars| {
                    self.normalize_rules.add_rule(&from_chars, &to_fons)
                })?;
            }
            _ => eprintln!("TODO: {:?}", rule),
        }
        Ok(())
    }

    pub fn try_get_alias(&self, alias: &str) -> Option<FonSet> {
        self.aliases.get(alias).cloned()
    }

    pub fn get_alias(&self, alias: &str) -> Result<FonSet> {
        self.try_get_alias(alias)
            .ok_or_else(|| NoSuchAlias(alias.into()))
    }

    pub fn set_alias(&mut self, name: String, fons: FonSet) {
        self.aliases.insert(name, fons);
    }

    fn resolve_rule_item_set(&mut self, items: &rulefile::ItemSet) -> Result<FonSet> {
        let mut fons = FonSet::new();
        for item in items {
            match item {
                rulefile::Item::Char(c) => fons |= self.fon_registry.add(*c)?,
                rulefile::Item::Alias(name) => fons |= self.get_alias(name)?,
                rulefile::Item::None => fons |= NO_FON,
            }
        }
        Ok(fons)
    }

    fn resolve_norm_rule_result_into(
        &mut self,
        rule_result: &rulefile::ItemSeq,
        normalized: &mut Vec<FonId>,
    ) -> Result<()> {
        if rule_result != &[rulefile::Item::None] {
            for item in rule_result {
                match item {
                    rulefile::Item::Char(c) => normalized.push(self.fon_registry.add(*c)?),
                    _ => return Err(InvalidNormRule(rulefile::item_seq_to_str(rule_result))),
                }
            }
        }
        Ok(())
    }

    fn resolve_norm_rule_result(&mut self, rule_result: &rulefile::ItemSeq) -> Result<Vec<FonId>> {
        let mut normalized = Vec::new();
        self.resolve_norm_rule_result_into(rule_result, &mut normalized)?;
        Ok(normalized)
    }

    fn resolve_norm_rule_lhs(&mut self, rule_lhs: &rulefile::ItemSetSeq) -> Result<Vec<Vec<char>>> {
        let fonsets: Result<Vec<FonSet>> = rule_lhs
            .iter()
            .map(|items| self.resolve_rule_item_set(items))
            .collect();
        let fonsets = fonsets?;
        if !FonSet::seq_is_valid(&fonsets) {
            return Err(InvalidNormRule(rulefile::item_set_seq_to_str(rule_lhs)));
        }

        let mut output = Vec::new();
        for fon_set in fonsets {
            let char_set: Result<Vec<char>> = fon_set
                .iter()
                .map(|i| self.fon_registry.get_fon(i))
                .collect();
            output.push(char_set?);
        }
        Ok(output)
    }

    pub fn add_to_dictionary(&mut self, word: &str, normalized: &[FonId]) -> Result<()> {
        match self.dictionary.get_mut(normalized) {
            Some(words) => {
                if words.iter().find(|w| w.as_ref() == word).is_none() {
                    words.push(word.into());
                }
            }
            None => {
                self.dictionary.insert(normalized.into(), vec![word.into()]);
            }
        }
        Ok(())
    }

    pub fn load_dictionary<R: BufRead>(&mut self, mut input: R) -> Result<()> {
        let mut line = String::new();
        let mut normalized = Vec::new();
        while input.read_line(&mut line)? > 0 {
            let word = line.trim();
            self.normalize_into(word, &mut normalized)?;
            self.add_to_dictionary(word, &normalized)?;
            line.clear();
            normalized.clear();
        }
        Ok(())
    }

    pub fn words_iter(&self, fonseq: &[FonId]) -> impl Iterator<Item = &str> {
        self.dictionary
            .get(fonseq)
            .into_iter()
            .flat_map(|v| v.iter().map(Box::borrow))
    }

    pub fn search(&self, word: &str) -> impl Iterator<Item = Result<(&str, Cost)>> {
        // initialize two optional iterators, depending on if we get a startup error
        let (startup_err, items) = match self.normalize(word) {
            Ok(normalized) => (None, Some(self.words_iter(&normalized).map(|w| Ok((w, 0))))),
            Err(e) => (Some(iter::once(Err(e))), None),
        };
        // chain them together, using into_iter().flatten() to either extract or turn into an empty iterator
        // this trick allows us to keep a consistent type of the final iterator
        // trick from Niko Matsakis https://stackoverflow.com/a/52064434
        startup_err
            .into_iter()
            .flatten()
            .chain(items.into_iter().flatten())
    }
}

impl RuleBasedNormalizer for BuscaCfg {
    fn rule_set(&self) -> &NormalizeRuleSet {
        &self.normalize_rules
    }

    fn fon_registry(&self) -> &FonRegistry {
        &self.fon_registry
    }
}

fn for_cartesian_product<T, S, F>(items: &[S], mut f: F) -> Result<()>
where
    S: AsRef<[T]>,
    T: Copy + Default,
    F: FnMut(&[T]) -> Result<()>,
{
    let len = items.len();
    let mut indices: Vec<usize> = vec![0; len];
    let mut output: Vec<T> = vec![Default::default(); len];
    loop {
        for i in 0..len {
            output[i] = items[i].as_ref()[indices[i]];
        }
        f(&output)?;
        for idx in (0..len).rev() {
            indices[idx] += 1;
            if indices[idx] == items[idx].as_ref().len() {
                if idx == 0 {
                    return Ok(());
                }
                indices[idx] = 0;
            } else {
                break;
            }
        }
    }
}
