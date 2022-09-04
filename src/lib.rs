use std::borrow::Borrow;
use std::cmp::min;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::io;
use std::io::BufRead;

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

    #[error("No such fon id {0:?}")]
    NoSuchFonId(u8),

    #[error("Already had a normalization rule for {0:?}")]
    DuplicateNormRule(Box<[char]>),

    #[error("Invalid normalization rule: {0}")]
    InvalidNormRule(String),

    #[error("No such alias {0:?}")]
    NoSuchAlias(String),

    #[error("Invalid replace rule")]
    InvalidReplaceRule,

    #[error("IO error {source:?}")]
    Io {
        #[from]
        source: io::Error,
    },

    #[error("Parsing error on line {line_no}: {text:?}")]
    ParseErr { line_no: usize, text: String },
}

use FonError::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
#[repr(transparent)]
pub struct Fon {
    id: u8,
}

const MAX_FON_ID: u8 = 127;

const MAX_FONS: usize = MAX_FON_ID as usize + 1;

pub const NO_FON: Fon = Fon { id: 0 };

pub const NO_FON_CHAR: char = '_';

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, PartialOrd, Ord, Hash)]
pub struct FonSet {
    bits: u128,
}

impl From<u8> for Fon {
    fn from(id: u8) -> Self {
        debug_assert!(id <= MAX_FON_ID);
        Fon { id }
    }
}

impl From<usize> for Fon {
    fn from(i: usize) -> Self {
        debug_assert!(i < MAX_FONS);
        Fon { id: i as u8 }
    }
}

impl FonSet {
    pub const EMPTY: FonSet = FonSet { bits: 0 };

    pub fn new() -> FonSet {
        FonSet::EMPTY
    }

    pub fn contains(&self, fon: Fon) -> bool {
        self.bits & (1 << fon.id) != 0
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
            fonset: *self,
            index: 0,
        }
    }

    pub fn fons(&self, reg: &FonRegistry) -> Result<Vec<char>> {
        let mut result: Vec<char> = Vec::new();
        for i in self {
            result.push(reg.get_fon_char(i)?);
        }
        Ok(result)
    }

    pub fn seq_is_empty<S: AsRef<[FonSet]> + ?Sized>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.is_empty() || slice.contains(&FonSet::EMPTY)
    }

    pub fn seq_is_real<S: AsRef<[FonSet]> + ?Sized>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.is_empty() || slice.iter().all(|s| s.is_real())
    }

    pub fn seq_is_valid<S: AsRef<[FonSet]> + ?Sized>(seq: &S) -> bool {
        let slice = seq.as_ref();
        slice.len() <= 1 || slice[1..slice.len() - 1].iter().all(|s| s.is_real())
    }

    fn first_fon(&self) -> Option<Fon> {
        self.iter().next()
    }

    fn next_fon_after(&self, mut fon: Fon) -> Option<Fon> {
        while fon.id < MAX_FON_ID {
            fon.id += 1;
            if self.contains(fon) {
                return Some(fon);
            }
        }
        None
    }

    pub fn seq_for_each_fon_seq<S, F>(seq: &S, mut action: F)
    where
        S: AsRef<[FonSet]> + ?Sized,
        F: FnMut(&[Fon]),
    {
        let slice = seq.as_ref();
        let mut buffer: Vec<Fon> = slice
            .iter()
            .map(|fonset| fonset.first_fon().unwrap())
            .collect();
        loop {
            action(&buffer);
            for i in (0..slice.len()).rev() {
                if let Some(fon) = slice[i].next_fon_after(buffer[i]) {
                    buffer[i] = fon;
                    break;
                } else if i == 0 {
                    return;
                } else {
                    buffer[i] = slice[i].first_fon().unwrap();
                }
            }
        }
    }

    pub fn seq_from_fonseq<S: IntoIterator<Item = Fon>>(seq: S) -> Vec<FonSet> {
        seq.into_iter().map(FonSet::from).collect()
    }

    fn seq_match_at_into<S: AsRef<[FonSet]>>(
        pattern: S,
        word: &[FonSet],
        word_idx: usize,
        result_buf: &mut Vec<FonSet>,
    ) -> bool {
        let pattern = pattern.as_ref();
        debug_assert!(word_idx + pattern.len() <= word.len());
        result_buf.clear();
        result_buf.extend_from_slice(word);
        for pattern_idx in 0..pattern.len() {
            result_buf[word_idx + pattern_idx] &= pattern[pattern_idx];
        }
        !FonSet::seq_is_empty(&result_buf[word_idx..word_idx + pattern.len()])
    }
}

impl From<Fon> for FonSet {
    fn from(fon: Fon) -> Self {
        FonSet { bits: 1 << fon.id }
    }
}

impl<const N: usize> From<[Fon; N]> for FonSet {
    fn from(fons: [Fon; N]) -> Self {
        let mut s = FonSet::new();
        for f in fons {
            s |= f;
        }
        s
    }
}

impl From<&[Fon]> for FonSet {
    fn from(fons: &[Fon]) -> Self {
        let mut s = FonSet::new();
        for f in fons {
            s |= *f;
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

impl std::ops::BitOr<Fon> for FonSet {
    type Output = Self;

    fn bitor(self, rhs: Fon) -> Self::Output {
        self | FonSet::from(rhs)
    }
}

impl std::ops::BitOrAssign for FonSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.bits |= rhs.bits;
    }
}

impl std::ops::BitOrAssign<Fon> for FonSet {
    fn bitor_assign(&mut self, rhs: Fon) {
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

impl std::ops::Sub<Fon> for FonSet {
    type Output = Self;

    fn sub(self, rhs: Fon) -> Self::Output {
        self - FonSet::from(rhs)
    }
}

impl std::ops::SubAssign for FonSet {
    fn sub_assign(&mut self, rhs: Self) {
        self.bits &= !rhs.bits;
    }
}

impl std::ops::SubAssign<Fon> for FonSet {
    fn sub_assign(&mut self, rhs: Fon) {
        *self -= FonSet::from(rhs)
    }
}

pub struct FonSetIter {
    fonset: FonSet,
    index: usize,
}

impl Iterator for FonSetIter {
    type Item = Fon;
    fn next(&mut self) -> Option<Fon> {
        if self.fonset.is_empty() {
            None
        } else {
            // skip all zeros, get nonzero index, then move one past
            let zeros = self.fonset.bits.trailing_zeros();
            self.index += zeros as usize;
            let item = Some(Fon {
                id: self.index as u8,
            });
            self.fonset.bits >>= zeros;
            self.fonset.bits >>= 1;
            self.index += 1;
            item
        }
    }
}

impl IntoIterator for FonSet {
    type Item = Fon;
    type IntoIter = FonSetIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for &FonSet {
    type Item = Fon;
    type IntoIter = FonSetIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<Fon> for FonSet {
    fn from_iter<I: IntoIterator<Item = Fon>>(iter: I) -> Self {
        let mut s = FonSet::new();
        for i in iter {
            s |= i;
        }
        s
    }
}

impl<'a> FromIterator<&'a Fon> for FonSet {
    fn from_iter<I: IntoIterator<Item = &'a Fon>>(iter: I) -> Self {
        let mut s = FonSet::new();
        for &i in iter {
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

    pub fn add(&mut self, fon: char) -> Result<Fon> {
        match self.try_get_fon(fon) {
            Some(id) => Ok(id),
            None => {
                let id = self.fons.len();
                if id < MAX_FONS {
                    self.fons.push(fon);
                    Ok(Fon::from(id))
                } else {
                    Err(NoMoreFonIds)
                }
            }
        }
    }

    pub fn try_get_fon(&self, c: char) -> Option<Fon> {
        self.fons.iter().position(|&f| f == c).map(Fon::from)
    }

    pub fn get_fon(&self, c: char) -> Result<Fon> {
        self.try_get_fon(c).ok_or(NoSuchFon(c))
    }

    pub fn try_get_fon_char(&self, fon: Fon) -> Option<char> {
        self.fons.get(fon.id as usize).cloned()
    }

    pub fn get_fon_char(&self, fon: Fon) -> Result<char> {
        self.try_get_fon_char(fon).ok_or(NoSuchFonId(fon.id))
    }

    pub fn mark(&self) -> FonRegistryMark {
        FonRegistryMark(self.fons.len())
    }

    pub fn revert(&mut self, mark: FonRegistryMark) {
        self.fons.truncate(mark.0);
    }
}

impl Default for FonRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FonRegistryMark(usize);

trait Normalizer {
    fn normalize_into(&self, word: &str, normalized: &mut Vec<Fon>) -> Result<()>;

    fn normalize(&self, word: &str) -> Result<Vec<Fon>> {
        let mut normalized = Vec::new();
        self.normalize_into(word, &mut normalized)?;
        Ok(normalized)
    }
}

fn prenormalized_chars(word: &str) -> impl Iterator<Item = char> + '_ {
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
    fn new() -> NormalizeRuleSet {
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

trait RuleBasedNormalizer {
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

trait MutationRule {
    type T;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::T>)>(
        &self,
        word: &[Self::T],
        word_idx: usize,
        action: F,
        result_buf: &mut Vec<Self::T>,
    );

    fn for_each_match_at<F: FnMut(&mut Vec<Self::T>)>(
        &self,
        word: &[Self::T],
        word_idx: usize,
        action: F,
    ) {
        self.for_each_match_at_using(word, word_idx, action, &mut Vec::new());
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::T>, usize)>(
        &self,
        word: &[Self::T],
        mut action: F,
        result_buf: &mut Vec<Self::T>,
    ) {
        for word_idx in 0..=word.len() {
            self.for_each_match_at_using(
                word,
                word_idx,
                |result_buf| action(result_buf, word_idx),
                result_buf,
            );
        }
    }

    fn for_each_match<F: FnMut(&mut Vec<Self::T>, usize)>(&self, word: &[Self::T], action: F) {
        self.for_each_match_using(word, action, &mut Vec::new());
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReplaceRule<M>
where
    M: MutationRule,
{
    matcher: M,
    remove_idx: usize,
    remove_len: usize,
    replace_with: Box<[M::T]>,
}

impl<M> ReplaceRule<M>
where
    M: MutationRule,
    M::T: Clone,
{
    fn get_remove_start(&self) -> usize {
        self.remove_idx
    }

    fn get_remove_end(&self) -> usize {
        self.remove_idx + self.remove_len
    }

    fn get_remove_len(&self) -> usize {
        self.remove_len
    }

    fn splice_into(&self, index: usize, result_buf: &mut Vec<M::T>) {
        result_buf.splice(
            (index + self.get_remove_start())..(index + self.get_remove_end()),
            self.replace_with.iter().cloned(),
        );
    }
}

impl<M> MutationRule for ReplaceRule<M>
where
    M: MutationRule,
    M::T: Clone,
{
    type T = M::T;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::T>)>(
        &self,
        word: &[Self::T],
        word_idx: usize,
        mut action: F,
        result_buf: &mut Vec<Self::T>,
    ) {
        self.matcher.for_each_match_at_using(
            word,
            word_idx,
            |result_buf| {
                self.splice_into(word_idx, result_buf);
                action(result_buf)
            },
            result_buf,
        );
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::T>, usize)>(
        &self,
        word: &[Self::T],
        mut action: F,
        result_buf: &mut Vec<Self::T>,
    ) {
        self.matcher.for_each_match_using(
            word,
            |result_buf, word_idx| {
                self.splice_into(word_idx, result_buf);
                action(result_buf, word_idx)
            },
            result_buf,
        );
    }
}

fn create_replace_rule(
    lookbehind: &[FonSet],
    old_pattern: &[FonSet],
    new_pattern: &[FonSet],
    lookahead: &[FonSet],
) -> Result<MutRule> {
    if !FonSet::seq_is_real(&new_pattern) {
        return Err(InvalidReplaceRule);
    }
    let mut pattern = Vec::new();
    pattern.extend_from_slice(lookbehind);
    pattern.extend_from_slice(old_pattern);
    pattern.extend_from_slice(lookahead);
    if pattern.is_empty() || !FonSet::seq_is_valid(&pattern) {
        return Err(InvalidReplaceRule);
    }
    Ok(ReplaceRule {
        matcher: pattern.into(),
        remove_idx: lookbehind.len(),
        remove_len: old_pattern.len(),
        replace_with: new_pattern.into(),
    })
}

impl<S: AsRef<[FonSet]>> MutationRule for S {
    type T = FonSet;

    fn for_each_match_at_using<F: FnMut(&mut Vec<FonSet>)>(
        &self,
        word: &[FonSet],
        word_idx: usize,
        mut action: F,
        result_buf: &mut Vec<FonSet>,
    ) {
        if self.as_ref().len() + word_idx <= word.len()
            && FonSet::seq_match_at_into(self, word, word_idx, result_buf)
        {
            action(result_buf);
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<FonSet>, usize)>(
        &self,
        word: &[FonSet],
        mut action: F,
        result_buf: &mut Vec<FonSet>,
    ) {
        let pattern = self.as_ref();
        let pattern_first = pattern[0];
        if pattern.len() <= word.len() {
            for word_idx in 0..=(word.len() - pattern.len()) {
                if !(word[word_idx] & pattern_first).is_empty()
                    && FonSet::seq_match_at_into(pattern, word, word_idx, result_buf)
                {
                    action(result_buf, word_idx);
                }
            }
        }
    }
}

pub type Cost = i32;

#[derive(Debug, Clone)]
struct ReplaceRuleSet<M: MutationRule> {
    rules: Vec<M>,
}

impl<M: MutationRule> ReplaceRuleSet<M> {
    fn new() -> ReplaceRuleSet<M> {
        ReplaceRuleSet { rules: Vec::new() }
    }

    fn add_rule(&mut self, rule: M) {
        self.rules.push(rule);
    }
}

impl<M: MutationRule> Default for ReplaceRuleSet<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: MutationRule> From<M> for ReplaceRuleSet<M> {
    fn from(rule: M) -> Self {
        let mut set = Self::new();
        set.add_rule(rule);
        set
    }
}

impl<M: MutationRule> MutationRule for ReplaceRuleSet<M> {
    type T = M::T;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::T>)>(
        &self,
        word: &[Self::T],
        word_idx: usize,
        mut action: F,
        result_buf: &mut Vec<Self::T>,
    ) {
        for rule in &self.rules {
            rule.for_each_match_at_using(word, word_idx, &mut action, result_buf);
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::T>, usize)>(
        &self,
        word: &[Self::T],
        mut action: F,
        result_buf: &mut Vec<Self::T>,
    ) {
        for rule in &self.rules {
            rule.for_each_match_using(word, &mut action, result_buf);
        }
    }
}

type MutRule = ReplaceRule<Box<[FonSet]>>;

#[derive(Debug, Clone)]
struct ReplaceRuleCostSet<M: MutationRule> {
    costs: Vec<Cost>,
    rules: Vec<ReplaceRuleSet<M>>,
}

impl<M: MutationRule> ReplaceRuleCostSet<M> {
    fn new() -> ReplaceRuleCostSet<M> {
        ReplaceRuleCostSet {
            costs: Vec::new(),
            rules: Vec::new(),
        }
    }

    fn add_cost_idx(&mut self, cost: Cost) -> usize {
        match self.costs.binary_search(&cost) {
            Ok(idx) => idx,
            Err(idx) => {
                self.costs.insert(idx, cost);
                self.rules.insert(idx, ReplaceRuleSet::new());
                idx
            }
        }
    }

    fn add_rule(&mut self, rule: M, cost: Cost) {
        let idx = self.add_cost_idx(cost);
        self.rules[idx].add_rule(rule);
    }

    fn is_empty(&self) -> bool {
        self.costs.is_empty()
    }
}

trait SliceSet<T> {
    fn add_slice(&mut self, slice: &[T]) -> bool;
    fn has_slice(&self, slice: &[T]) -> bool;
}

impl<T: Ord + Copy> SliceSet<T> for BTreeSet<Box<[T]>> {
    fn add_slice(&mut self, slice: &[T]) -> bool {
        let is_new = !self.has_slice(slice);
        if is_new {
            self.insert(slice.into());
        }
        is_new
    }

    fn has_slice(&self, slice: &[T]) -> bool {
        self.contains(slice)
    }
}

#[derive(Debug, Clone)]
pub struct BuscaCfg {
    fon_registry: FonRegistry,
    dictionary: BTreeMap<Box<[Fon]>, Vec<Box<str>>>,
    normalize_rules: NormalizeRuleSet,
    aliases: BTreeMap<String, FonSet>,
    mutation_rules: ReplaceRuleCostSet<MutRule>,
}

impl BuscaCfg {
    pub fn new() -> BuscaCfg {
        BuscaCfg {
            fon_registry: FonRegistry::new(),
            dictionary: BTreeMap::new(),
            normalize_rules: NormalizeRuleSet::new(),
            aliases: BTreeMap::new(),
            mutation_rules: ReplaceRuleCostSet::new(),
        }
    }

    pub fn load_rules<R: BufRead>(&mut self, input: R) -> Result<()> {
        for (line_no, line) in input.lines().enumerate() {
            match rulefile::rule_line(&line?).finish() {
                Ok((_, Some(rule))) => self.add_rule(rule),
                Ok((_, None)) => Ok(()),
                Err(parse_err) => Err(ParseErr {
                    line_no: line_no + 1,
                    text: parse_err.input.to_owned(),
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
                    self.normalize_rules.add_rule(from_chars, &to_fons)
                })?;
            }
            Rule::Mut {
                cost,
                before,
                from,
                to,
                after,
            } => {
                let lookbehind = self.resolve_lookaround_item_set_seq(&before)?;
                let from = self.resolve_mutation_item_set_seq(&from)?;
                let to = self.resolve_mutation_item_set_seq(&to)?;
                let lookahead = self.resolve_lookaround_item_set_seq(&after)?;
                self.mutation_rules.add_rule(
                    create_replace_rule(&lookbehind, &from, &to, &lookahead)?,
                    cost,
                );
                self.mutation_rules.add_rule(
                    create_replace_rule(&lookbehind, &to, &from, &lookahead)?,
                    cost,
                );
            }
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

    fn resolve_norm_rule_result(&mut self, rule_result: &rulefile::ItemSeq) -> Result<Vec<Fon>> {
        let mut normalized = Vec::new();
        if rule_result != &[rulefile::Item::None] {
            for item in rule_result {
                match item {
                    rulefile::Item::Char(c) => normalized.push(self.fon_registry.add(*c)?),
                    _ => return Err(InvalidNormRule(rulefile::item_seq_to_str(rule_result))),
                }
            }
        }
        Ok(normalized)
    }

    fn resolve_rule_item_set_seq(
        &mut self,
        item_sets: &rulefile::ItemSetSeq,
    ) -> Result<Vec<FonSet>> {
        let mut result = Vec::new();
        for item_set in item_sets {
            result.push(self.resolve_rule_item_set(item_set)?);
        }
        Ok(result)
    }

    fn resolve_lookaround_item_set_seq(
        &mut self,
        item_sets: &rulefile::ItemSetSeq,
    ) -> Result<Vec<FonSet>> {
        self.resolve_rule_item_set_seq(item_sets)
    }

    fn resolve_mutation_item_set_seq(
        &mut self,
        item_sets: &rulefile::ItemSetSeq,
    ) -> Result<Vec<FonSet>> {
        let mut result = self.resolve_rule_item_set_seq(item_sets)?;
        if result.len() == 1 && result[0] == FonSet::from(NO_FON) {
            result.clear();
        } else if !FonSet::seq_is_real(&result) {
            return Err(InvalidReplaceRule);
        }
        Ok(result)
    }

    fn resolve_norm_rule_lhs(&mut self, rule_lhs: &rulefile::ItemSetSeq) -> Result<Vec<Vec<char>>> {
        let mark = self.fon_registry.mark();
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
                .map(|i| self.fon_registry.get_fon_char(i))
                .collect();
            output.push(char_set?);
        }
        self.fon_registry.revert(mark);
        Ok(output)
    }

    pub fn add_to_dictionary(&mut self, word: &str, normalized: &[Fon]) -> Result<()> {
        match self.dictionary.get_mut(normalized) {
            Some(words) => {
                if !words.iter().any(|w| w.as_ref() == word) {
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

    pub fn words_iter(&self, fonseq: &[Fon]) -> impl Iterator<Item = &str> {
        self.dictionary
            .get(fonseq)
            .into_iter()
            .flat_map(|v| v.iter().map(Box::borrow))
    }

    pub fn search(&self, word: &str) -> Result<Busca> {
        Ok(Busca::new(
            self,
            &FonSet::seq_from_fonseq(self.normalize(word)?),
        ))
    }
}

impl Default for BuscaCfg {
    fn default() -> Self {
        Self::new()
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct BuscaNode {
    total_cost: Cost,
    cost_idx: usize,
    word: Box<[FonSet]>,
}

impl BuscaNode {
    fn new(cfg: &BuscaCfg, word: &[FonSet], cost: Cost) -> BuscaNode {
        BuscaNode {
            total_cost: cost + cfg.mutation_rules.costs[0],
            cost_idx: 0,
            word: word.into(),
        }
    }

    fn inc_cost(&mut self, cfg: &BuscaCfg) -> Option<Cost> {
        let costs = &cfg.mutation_rules.costs;
        let new_cost_idx = self.cost_idx + 1;
        if new_cost_idx < costs.len() {
            self.total_cost = self.total_cost - costs[self.cost_idx] + costs[new_cost_idx];
            self.cost_idx = new_cost_idx;
            Some(self.total_cost)
        } else {
            None
        }
    }
}

impl Ord for BuscaNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.total_cost.cmp(&self.total_cost)
    }
}

impl PartialOrd for BuscaNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct Busca<'a> {
    cfg: &'a BuscaCfg,
    already_visited: BTreeSet<Box<[FonSet]>>,
    to_output: Vec<(&'a str, Cost)>,
    already_output: BTreeSet<&'a str>,
    current_node: Option<BuscaNode>,
    later_nodes: BinaryHeap<BuscaNode>,
}

impl<'a> Busca<'a> {
    fn new(cfg: &'a BuscaCfg, word: &[FonSet]) -> Busca<'a> {
        let mut busca = Busca {
            cfg,
            current_node: Some(BuscaNode::new(cfg, word, 0)),
            already_visited: BTreeSet::new(),
            already_output: BTreeSet::new(),
            to_output: Vec::new(),
            later_nodes: BinaryHeap::new(),
        };
        busca.visit_fonsetseq(word, 0);
        busca
    }

    fn search_current(&mut self) {
        let mut current_node = self.current_node.take().unwrap();
        self.cfg.mutation_rules.rules[current_node.cost_idx]
            .for_each_match(&*current_node.word, |result, _| {
                self.add_node(result, current_node.total_cost)
            });
        if let Some(new_cost) = current_node.inc_cost(self.cfg) {
            if let Some(lowest_later_node) = self.later_nodes.peek() {
                if new_cost > lowest_later_node.total_cost {
                    self.current_node = self.later_nodes.pop();
                    self.later_nodes.push(current_node);
                } else {
                    self.current_node = Some(current_node);
                }
            } else {
                self.current_node = Some(current_node);
            }
        } else {
            self.current_node = self.later_nodes.pop();
        }
    }

    fn visit_fonsetseq(&mut self, word_fonsetseq: &[FonSet], cost: Cost) -> bool {
        let is_new = self.already_visited.add_slice(word_fonsetseq);
        if is_new {
            FonSet::seq_for_each_fon_seq(word_fonsetseq, |word_fonseq| {
                self.visit_fonseq(word_fonseq, cost)
            });
        }
        is_new
    }

    fn visit_fonseq(&mut self, word_fonseq: &[Fon], cost: Cost) {
        for word_str in self.cfg.words_iter(word_fonseq) {
            self.visit_str(word_str, cost);
        }
    }

    fn visit_str(&mut self, word_str: &'a str, cost: Cost) {
        if self.already_output.insert(word_str) {
            self.to_output.push((word_str, cost));
        }
    }

    fn add_node(&mut self, word_fonsetseq: &[FonSet], cost: Cost) {
        if self.visit_fonsetseq(word_fonsetseq, cost) {
            self.later_nodes
                .push(BuscaNode::new(self.cfg, word_fonsetseq, cost));
        }
    }
}

impl<'a> Iterator for Busca<'a> {
    type Item = Option<(&'a str, Cost)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.to_output.is_empty() {
            if self.current_node.is_none() {
                return None;
            }
            self.search_current();
        }
        Some(self.to_output.pop())
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
