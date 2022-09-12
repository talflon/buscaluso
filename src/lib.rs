mod fon;
mod normalize;
mod rulefile;
mod rules;

#[cfg(test)]
mod tests;

use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt::Debug;
use std::io;
use std::io::BufRead;

use nom::Finish;
use rulefile::Rule;
use thiserror::Error;

use fon::*;
use normalize::*;
use rules::*;

pub type Result<T> = std::result::Result<T, FonError>;

#[derive(Error, Debug)]
pub enum FonError {
    #[error("No such fon {0:?}")]
    NoSuchFon(char),

    #[error("Ran out of fon ids")]
    NoMoreFonIds,

    #[error("No such fon id {0:?}")]
    NoSuchFonId(u64),

    #[error("Already had a normalization rule for {0:?}")]
    DuplicateNormRule(Box<[char]>),

    #[error("Invalid normalization rule: {0}")]
    InvalidNormRule(String),

    #[error("No such alias {0:?}")]
    NoSuchAlias(String),

    #[error("Invalid replace rule: {0}")]
    InvalidReplaceRule(String),

    #[error("IO error {source:?}")]
    Io {
        #[from]
        source: io::Error,
    },

    #[error("Parsing error on line {line_no}: {text:?}")]
    ParseErr { line_no: usize, text: String },
}

use FonError::*;

pub type Cost = u32;

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
    dictionary: BTreeDictionary,
    normalize_rules: NormalizeRuleSet,
    aliases: BTreeMap<String, FonSet>,
    mutation_rules: ReplaceRuleCostSet<MutRule>,
}

impl BuscaCfg {
    pub fn new() -> BuscaCfg {
        BuscaCfg {
            fon_registry: FonRegistry::new(),
            dictionary: BTreeDictionary::new(),
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
                add_replace_rules(
                    &mut self.mutation_rules,
                    &lookbehind,
                    &from,
                    &to,
                    &lookahead,
                    cost,
                )?;
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
        } else if !result.is_real_seq() {
            return Err(InvalidReplaceRule("portion not \"real\"".into()));
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
        if !fonsets.is_valid_seq() {
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

    pub fn load_dictionary<R: BufRead>(&mut self, mut input: R) -> Result<()> {
        let mut line = String::new();
        let mut normalized = Vec::new();
        while input.read_line(&mut line)? > 0 {
            let word = line.trim();
            self.normalize_into(word, &mut normalized)?;
            self.dictionary.add_word(&normalized, word);
            line.clear();
            normalized.clear();
        }
        Ok(())
    }

    pub fn search(&self, word: &str) -> Result<Busca> {
        Ok(Busca::new(
            self,
            &fonsetseq_from_fonseq(self.normalize(word)?),
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
        busca.add_node(word, 0, &mut Vec::new());
        busca
    }

    fn search_current(&mut self, fonset_buffer: &mut Vec<FonSet>, fon_buffer: &mut Vec<Fon>) {
        if let Some(mut current_node) = self.nodes.pop() {
            self.cfg.mutation_rules.rules[current_node.cost_idx].for_each_match(
                &*current_node.word,
                |result, _| {
                    if !result.is_subset_of_seq(&current_node.word) {
                        self.add_node(result, current_node.total_cost, fon_buffer)
                    }
                },
                fonset_buffer,
            );
            if current_node.inc_cost(self.cfg).is_some() {
                self.nodes.push(current_node);
            }
        };
    }

    fn visit_fonsetseq(
        &mut self,
        word_fonsetseq: &[FonSet],
        cost: Cost,
        buffer: &mut Vec<Fon>,
    ) -> bool {
        let is_new = self.already_visited.add_slice(word_fonsetseq);
        if is_new {
            self.cfg
                .dictionary
                .for_each_word_matching(word_fonsetseq, buffer, |word_str| {
                    self.visit_str(word_str, cost)
                });
        }
        is_new
    }

    fn visit_str(&mut self, word_str: &'a str, cost: Cost) {
        if self.already_output.insert(word_str) {
            self.to_output.push((word_str, cost));
        }
    }

    fn add_node(&mut self, word_fonsetseq: &[FonSet], cost: Cost, buffer: &mut Vec<Fon>) {
        if self.visit_fonsetseq(word_fonsetseq, cost, buffer) {
            self.nodes
                .push(BuscaNode::new(self.cfg, word_fonsetseq, cost));
        }
    }

    pub fn iter<'b>(&'b mut self) -> BuscaIter<'a, 'b> {
        BuscaIter {
            busca: self,
            fonset_buffer: Vec::new(),
            fon_buffer: Vec::new(),
        }
    }

    fn try_next(
        &mut self,
        fonset_buffer: &mut Vec<FonSet>,
        fon_buffer: &mut Vec<Fon>,
    ) -> Option<Option<(&'a str, Cost)>> {
        if self.to_output.is_empty() {
            if self.nodes.is_empty() {
                return None;
            }
            self.search_current(fonset_buffer, fon_buffer);
        }
        Some(self.to_output.pop())
    }
}

#[derive(Debug)]
pub struct BuscaIter<'a, 'b> {
    busca: &'b mut Busca<'a>,
    fonset_buffer: Vec<FonSet>,
    fon_buffer: Vec<Fon>,
}

impl<'a, 'b> Iterator for BuscaIter<'a, 'b> {
    type Item = Option<(&'a str, Cost)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.busca
            .try_next(&mut self.fonset_buffer, &mut self.fon_buffer)
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

#[derive(Debug, Clone)]
struct BTreeDictionary {
    words: BTreeMap<Box<[Fon]>, Vec<Box<str>>>,
}

impl BTreeDictionary {
    fn new() -> Self {
        BTreeDictionary {
            words: BTreeMap::new(),
        }
    }

    fn add_word(&mut self, normalized: &[Fon], word: &str) {
        match self.words.get_mut(normalized) {
            Some(words) => {
                if !words.iter().any(|w| w.as_ref() == word) {
                    words.push(word.into());
                }
            }
            None => {
                self.words.insert(normalized.into(), vec![word.into()]);
            }
        }
    }

    fn for_each_word_normalized<'a, F>(&'a self, fonseq: &[Fon], action: F)
    where
        F: FnMut(&'a str),
    {
        self.words
            .get(fonseq)
            .iter()
            .flat_map(|v| v.iter())
            .map(Borrow::borrow)
            .for_each(action);
    }

    fn for_each_word_matching<'a>(
        &'a self,
        fonsetseq: &[FonSet],
        buffer: &mut Vec<Fon>,
        mut action: impl FnMut(&'a str),
    ) {
        fonsetseq.for_each_fon_seq(buffer, |fonseq| {
            self.for_each_word_normalized(fonseq, &mut action)
        });
    }
}
