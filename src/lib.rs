mod fon;
mod rulefile;

#[cfg(test)]
mod tests;

use std::borrow::Borrow;
use std::cmp::min;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt::Debug;
use std::io;
use std::io::BufRead;

use nom::Finish;
use rulefile::Rule;
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use fon::*;

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
    type Alph;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: F,
        result_buf: &mut Vec<Self::Alph>,
    );

    fn for_each_match_at<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: F,
    ) {
        self.for_each_match_at_using(word, word_idx, action, &mut Vec::new());
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        word: &[Self::Alph],
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
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

    fn for_each_match<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        word: &[Self::Alph],
        action: F,
    ) {
        self.for_each_match_using(word, action, &mut Vec::new());
    }
}

trait FixedLenRule
where
    Self: MutationRule,
{
    fn match_len(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReplaceRule<M>
where
    M: MutationRule,
{
    matcher: M,
    remove_idx: usize,
    remove_len: usize,
    replace_with: Box<[M::Alph]>,
}

impl<M> ReplaceRule<M>
where
    M: MutationRule,
    M::Alph: Clone,
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

    fn splice_into(&self, index: usize, result_buf: &mut Vec<M::Alph>) {
        result_buf.splice(
            (index + self.get_remove_start())..(index + self.get_remove_end()),
            self.replace_with.iter().cloned(),
        );
    }
}

impl<M> MutationRule for ReplaceRule<M>
where
    M: MutationRule,
    M::Alph: Clone,
{
    type Alph = M::Alph;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
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

    fn for_each_match_using<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        word: &[Self::Alph],
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
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

impl<M> FixedLenRule for ReplaceRule<M>
where
    M: FixedLenRule,
    M::Alph: Clone,
{
    fn match_len(&self) -> usize {
        self.matcher.match_len()
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
    type Alph = FonSet;

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

impl<S: AsRef<[FonSet]>> FixedLenRule for S {
    fn match_len(&self) -> usize {
        self.as_ref().len()
    }
}

#[derive(Clone, Debug)]
struct StartAnchoredRule<M: MutationRule>(M);

impl<M: MutationRule> MutationRule for StartAnchoredRule<M> {
    type Alph = M::Alph;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if word_idx == 0 {
            self.0
                .for_each_match_at_using(word, word_idx, action, result_buf);
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        word: &[Self::Alph],
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        self.0
            .for_each_match_at_using(word, 0, |result_buf| action(result_buf, 0), result_buf);
    }
}

impl<M> FixedLenRule for StartAnchoredRule<M>
where
    M: FixedLenRule,
{
    fn match_len(&self) -> usize {
        self.0.match_len()
    }
}

#[derive(Clone, Debug)]
struct EndAnchoredRule<M: FixedLenRule>(M);

impl<M: FixedLenRule> MutationRule for EndAnchoredRule<M> {
    type Alph = M::Alph;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if word_idx + self.match_len() == word.len() {
            self.0
                .for_each_match_at_using(word, word_idx, action, result_buf);
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        word: &[Self::Alph],
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if let Some(word_idx) = word.len().checked_sub(self.match_len()) {
            self.0.for_each_match_at_using(
                word,
                word_idx,
                |result_buf| action(result_buf, word_idx),
                result_buf,
            );
        }
    }
}

impl<M> FixedLenRule for EndAnchoredRule<M>
where
    M: FixedLenRule,
{
    fn match_len(&self) -> usize {
        self.0.match_len()
    }
}

#[derive(Clone, Debug)]
struct BothAnchoredRule<M: FixedLenRule>(M);

impl<M: FixedLenRule> MutationRule for BothAnchoredRule<M> {
    type Alph = M::Alph;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if word_idx == 0 && self.match_len() == word.len() {
            self.0
                .for_each_match_at_using(word, word_idx, action, result_buf);
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        word: &[Self::Alph],
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if self.match_len() == word.len() {
            self.0
                .for_each_match_at_using(word, 0, |result_buf| action(result_buf, 0), result_buf);
        }
    }
}

impl<M> FixedLenRule for BothAnchoredRule<M>
where
    M: FixedLenRule,
{
    fn match_len(&self) -> usize {
        self.0.match_len()
    }
}

pub type Cost = u32;

#[derive(Debug, Clone)]
struct ReplaceRuleSet<M: FixedLenRule> {
    any_rules: Vec<M>,
    start_rules: Vec<StartAnchoredRule<M>>,
    end_rules: Vec<EndAnchoredRule<M>>,
    both_rules: Vec<BothAnchoredRule<M>>,
}

impl<M: FixedLenRule> ReplaceRuleSet<M> {
    fn new() -> ReplaceRuleSet<M> {
        ReplaceRuleSet {
            any_rules: Vec::new(),
            start_rules: Vec::new(),
            end_rules: Vec::new(),
            both_rules: Vec::new(),
        }
    }

    fn add_any_rule(&mut self, rule: M) {
        self.any_rules.push(rule);
    }
}

impl<M: FixedLenRule> Default for ReplaceRuleSet<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: FixedLenRule> MutationRule for ReplaceRuleSet<M> {
    type Alph = M::Alph;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::Alph>)>(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for rule in &self.any_rules {
            rule.for_each_match_at_using(word, word_idx, &mut action, result_buf);
        }
        for rule in &self.end_rules {
            rule.for_each_match_at_using(word, word_idx, &mut action, result_buf);
        }
        if word_idx == 0 {
            for rule in &self.start_rules {
                rule.for_each_match_at_using(word, word_idx, &mut action, result_buf);
            }
            for rule in &self.both_rules {
                rule.for_each_match_at_using(word, word_idx, &mut action, result_buf);
            }
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::Alph>, usize)>(
        &self,
        word: &[Self::Alph],
        mut action: F,
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for rule in &self.any_rules {
            rule.for_each_match_using(word, &mut action, result_buf);
        }
        for rule in &self.start_rules {
            rule.for_each_match_using(word, &mut action, result_buf);
        }
        for rule in &self.end_rules {
            rule.for_each_match_using(word, &mut action, result_buf);
        }
        for rule in &self.both_rules {
            rule.for_each_match_using(word, &mut action, result_buf);
        }
    }
}

type MutRule = ReplaceRule<Box<[FonSet]>>;

#[derive(Debug, Clone)]
struct ReplaceRuleCostSet<M: FixedLenRule> {
    costs: Vec<Cost>,
    rules: Vec<ReplaceRuleSet<M>>,
}

impl<M: FixedLenRule> ReplaceRuleCostSet<M> {
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

    fn add_any_rule(&mut self, rule: M, cost: Cost) {
        let idx = self.add_cost_idx(cost);
        self.rules[idx].add_any_rule(rule);
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
                self.mutation_rules.add_any_rule(
                    create_replace_rule(&lookbehind, &from, &to, &lookahead)?,
                    cost,
                );
                self.mutation_rules.add_any_rule(
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
    nodes: BinaryHeap<BuscaNode>,
}

impl<'a> Busca<'a> {
    fn new(cfg: &'a BuscaCfg, word: &[FonSet]) -> Busca<'a> {
        let mut busca = Busca {
            cfg,
            already_visited: BTreeSet::new(),
            already_output: BTreeSet::new(),
            to_output: Vec::new(),
            nodes: BinaryHeap::new(),
        };
        busca.add_node(word, 0);
        busca
    }

    fn search_current(&mut self) {
        if let Some(mut current_node) = self.nodes.pop() {
            self.cfg.mutation_rules.rules[current_node.cost_idx]
                .for_each_match(&*current_node.word, |result, _| {
                    self.add_node(result, current_node.total_cost)
                });
            if current_node.inc_cost(self.cfg).is_some() {
                self.nodes.push(current_node);
            }
        };
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
            self.nodes
                .push(BuscaNode::new(self.cfg, word_fonsetseq, cost));
        }
    }
}

impl<'a> Iterator for Busca<'a> {
    type Item = Option<(&'a str, Cost)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.to_output.is_empty() {
            if self.nodes.is_empty() {
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
