// Copyright © 2022 Daniel Getz
// SPDX-License-Identifier: MIT

#[cfg(test)]
mod tests;

use crate::FonError::*;
use crate::{Cost, FonSet, FonSetSeq, Result, NO_FON};

pub type MutRule = ReplaceRule<Box<[FonSet]>>;

pub fn create_replace_rule(
    lookbehind: &[FonSet],
    old_pattern: &[FonSet],
    new_pattern: &[FonSet],
    lookahead: &[FonSet],
) -> Result<MutRule> {
    let mut pattern = Vec::new();
    pattern.extend_from_slice(lookbehind);
    pattern.extend_from_slice(old_pattern);
    pattern.extend_from_slice(lookahead);
    if pattern.is_empty_seq() {
        return Err(InvalidReplaceRule(format!(
            "pattern empty: {:?} | {:?} > {:?} | {:?}",
            lookbehind, old_pattern, new_pattern, lookahead,
        )));
    }
    Ok(ReplaceRule {
        matcher: pattern.into(),
        remove_idx: lookbehind.len(),
        remove_len: old_pattern.len(),
        replace_with: new_pattern.into(),
    })
}

pub fn add_replace_rules(
    rule_set: &mut ReplaceRuleCostSet<MutRule>,
    lookbehind: &[FonSet],
    old_pattern: &[FonSet],
    new_pattern: &[FonSet],
    lookahead: &[FonSet],
    cost: Cost,
) -> Result<()> {
    rule_set.add_any_rule(
        create_replace_rule(lookbehind, old_pattern, new_pattern, lookahead)?,
        cost,
    );
    rule_set.add_any_rule(
        create_replace_rule(lookbehind, new_pattern, old_pattern, lookahead)?,
        cost,
    );
    if let Some(s) = lookbehind.first() {
        if s.contains(NO_FON) {
            let lookbehind = &lookbehind[1..];
            rule_set.add_start_rule(
                StartAnchoredRule(create_replace_rule(
                    lookbehind,
                    new_pattern,
                    old_pattern,
                    lookahead,
                )?),
                cost,
            );
            rule_set.add_start_rule(
                StartAnchoredRule(create_replace_rule(
                    lookbehind,
                    old_pattern,
                    new_pattern,
                    lookahead,
                )?),
                cost,
            );
            if let Some(s) = lookahead.last() {
                if s.contains(NO_FON) {
                    let lookahead = &lookahead[0..lookahead.len() - 1];
                    rule_set.add_both_rule(
                        BothAnchoredRule(create_replace_rule(
                            lookbehind,
                            new_pattern,
                            old_pattern,
                            lookahead,
                        )?),
                        cost,
                    );
                    rule_set.add_both_rule(
                        BothAnchoredRule(create_replace_rule(
                            lookbehind,
                            old_pattern,
                            new_pattern,
                            lookahead,
                        )?),
                        cost,
                    );
                }
            }
        }
    }
    if let Some(s) = lookahead.last() {
        if s.contains(NO_FON) {
            let lookahead = &lookahead[0..lookahead.len() - 1];
            rule_set.add_end_rule(
                EndAnchoredRule(create_replace_rule(
                    lookbehind,
                    new_pattern,
                    old_pattern,
                    lookahead,
                )?),
                cost,
            );
            rule_set.add_end_rule(
                EndAnchoredRule(create_replace_rule(
                    lookbehind,
                    old_pattern,
                    new_pattern,
                    lookahead,
                )?),
                cost,
            );
        }
    }
    Ok(())
}

pub trait MutationRule {
    type Alph;

    fn for_each_match_at(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    );

    fn for_each_match(
        &self,
        word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for word_idx in 0..=word.len() {
            self.for_each_match_at(
                word,
                word_idx,
                |result_buf| action(result_buf, word_idx),
                result_buf,
            );
        }
    }

    // fn matches_at(&self, word: &[Self::Alph], word_idx: usize) -> bool;
}

pub trait FixedLenRule
where
    Self: MutationRule,
{
    fn match_len(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplaceRule<M>
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
    pub fn get_remove_start(&self) -> usize {
        self.remove_idx
    }

    pub fn get_remove_end(&self) -> usize {
        self.remove_idx + self.remove_len
    }

    pub fn get_remove_len(&self) -> usize {
        self.remove_len
    }

    pub fn splice_into(&self, index: usize, result_buf: &mut Vec<M::Alph>) {
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

    fn for_each_match_at(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        mut action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        self.matcher.for_each_match_at(
            word,
            word_idx,
            |result_buf| {
                self.splice_into(word_idx, result_buf);
                action(result_buf)
            },
            result_buf,
        );
    }

    fn for_each_match(
        &self,
        word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        self.matcher.for_each_match(
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

impl<S: FonSetSeq> MutationRule for S {
    type Alph = FonSet;

    fn for_each_match_at(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        mut action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if self.as_ref().len() + word_idx <= word.len() && self.match_at(word, word_idx, result_buf)
        {
            action(result_buf)
        }
    }

    fn for_each_match(
        &self,
        word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        let pattern = self.as_ref();
        let pattern_first = pattern[0];
        if pattern.len() <= word.len() {
            for word_idx in 0..=(word.len() - pattern.len()) {
                if !(word[word_idx] & pattern_first).is_empty()
                    && self.match_at(word, word_idx, result_buf)
                {
                    action(result_buf, word_idx);
                }
            }
        }
    }
}

impl<S: FonSetSeq> FixedLenRule for S {
    fn match_len(&self) -> usize {
        self.as_ref().len()
    }
}

#[derive(Clone, Debug)]
pub struct StartAnchoredRule<M: MutationRule>(pub M);

impl<M: MutationRule> MutationRule for StartAnchoredRule<M> {
    type Alph = M::Alph;

    fn for_each_match_at(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if word_idx == 0 {
            self.0.for_each_match_at(word, word_idx, action, result_buf);
        }
    }

    fn for_each_match(
        &self,
        word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        self.0
            .for_each_match_at(word, 0, |result_buf| action(result_buf, 0), result_buf);
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
pub struct EndAnchoredRule<M: FixedLenRule>(pub M);

impl<M: FixedLenRule> MutationRule for EndAnchoredRule<M> {
    type Alph = M::Alph;

    fn for_each_match_at(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if word_idx + self.match_len() == word.len() {
            self.0.for_each_match_at(word, word_idx, action, result_buf);
        }
    }

    fn for_each_match(
        &self,
        word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if let Some(word_idx) = word.len().checked_sub(self.match_len()) {
            self.0.for_each_match_at(
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
pub struct BothAnchoredRule<M: FixedLenRule>(pub M);

impl<M: FixedLenRule> MutationRule for BothAnchoredRule<M> {
    type Alph = M::Alph;

    fn for_each_match_at(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if word_idx == 0 && self.match_len() == word.len() {
            self.0.for_each_match_at(word, word_idx, action, result_buf);
        }
    }

    fn for_each_match(
        &self,
        word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        if self.match_len() == word.len() {
            self.0
                .for_each_match_at(word, 0, |result_buf| action(result_buf, 0), result_buf);
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

#[derive(Debug, Clone)]
pub struct ReplaceRuleSet<M: FixedLenRule> {
    any_rules: Vec<M>,
    start_rules: Vec<StartAnchoredRule<M>>,
    end_rules: Vec<EndAnchoredRule<M>>,
    both_rules: Vec<BothAnchoredRule<M>>,
}

impl<M: FixedLenRule> ReplaceRuleSet<M> {
    pub fn new() -> ReplaceRuleSet<M> {
        ReplaceRuleSet {
            any_rules: Vec::new(),
            start_rules: Vec::new(),
            end_rules: Vec::new(),
            both_rules: Vec::new(),
        }
    }

    pub fn add_any_rule(&mut self, rule: M) {
        self.any_rules.push(rule);
    }

    pub fn add_start_rule(&mut self, rule: StartAnchoredRule<M>) {
        self.start_rules.push(rule);
    }

    pub fn add_end_rule(&mut self, rule: EndAnchoredRule<M>) {
        self.end_rules.push(rule);
    }

    pub fn add_both_rule(&mut self, rule: BothAnchoredRule<M>) {
        self.both_rules.push(rule);
    }
}

impl<M: FixedLenRule> Default for ReplaceRuleSet<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: FixedLenRule> MutationRule for ReplaceRuleSet<M> {
    type Alph = M::Alph;

    fn for_each_match_at(
        &self,
        word: &[Self::Alph],
        word_idx: usize,
        mut action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for rule in &self.any_rules {
            rule.for_each_match_at(word, word_idx, &mut action, result_buf);
        }
        for rule in &self.end_rules {
            rule.for_each_match_at(word, word_idx, &mut action, result_buf);
        }
        if word_idx == 0 {
            for rule in &self.start_rules {
                rule.for_each_match_at(word, 0, &mut action, result_buf);
            }
            for rule in &self.both_rules {
                rule.for_each_match_at(word, 0, &mut action, result_buf);
            }
        }
    }

    fn for_each_match(
        &self,
        word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for rule in &self.any_rules {
            rule.for_each_match(word, &mut action, result_buf);
        }
        for rule in &self.start_rules {
            rule.for_each_match(word, &mut action, result_buf);
        }
        for rule in &self.end_rules {
            rule.for_each_match(word, &mut action, result_buf);
        }
        for rule in &self.both_rules {
            rule.for_each_match(word, &mut action, result_buf);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReplaceRuleCostSet<M: FixedLenRule> {
    pub costs: Vec<Cost>,
    pub rules: Vec<ReplaceRuleSet<M>>,
}

impl<M: FixedLenRule> ReplaceRuleCostSet<M> {
    pub fn new() -> ReplaceRuleCostSet<M> {
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

    pub fn add_any_rule(&mut self, rule: M, cost: Cost) {
        let idx = self.add_cost_idx(cost);
        self.rules[idx].add_any_rule(rule);
    }

    pub fn add_start_rule(&mut self, rule: StartAnchoredRule<M>, cost: Cost) {
        let idx = self.add_cost_idx(cost);
        self.rules[idx].add_start_rule(rule);
    }

    pub fn add_end_rule(&mut self, rule: EndAnchoredRule<M>, cost: Cost) {
        let idx = self.add_cost_idx(cost);
        self.rules[idx].add_end_rule(rule);
    }

    pub fn add_both_rule(&mut self, rule: BothAnchoredRule<M>, cost: Cost) {
        let idx = self.add_cost_idx(cost);
        self.rules[idx].add_both_rule(rule);
    }

    pub fn is_empty(&self) -> bool {
        self.costs.is_empty()
    }
}
