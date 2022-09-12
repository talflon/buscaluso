use quickcheck::{Arbitrary, TestResult};
use quickcheck_macros::*;

use super::*;
use crate::fon::tests::*;
use crate::tests::*;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct MockMutationRule<T> {
    matches: Vec<(Vec<T>, usize)>,
    matches_at: Vec<Vec<T>>,
    match_len: usize,
}

impl<T: Clone> MutationRule for MockMutationRule<T> {
    type Alph = T;

    fn for_each_match_at(
        &self,
        _word: &[Self::Alph],
        _word_idx: usize,
        mut action: impl FnMut(&mut Vec<Self::Alph>),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for result in &self.matches_at {
            result_buf.clear();
            result_buf.extend_from_slice(result);
            action(result_buf);
        }
    }

    fn for_each_match(
        &self,
        _word: &[Self::Alph],
        mut action: impl FnMut(&mut Vec<Self::Alph>, usize),
        result_buf: &mut Vec<Self::Alph>,
    ) {
        for (result, idx) in &self.matches {
            result_buf.clear();
            result_buf.extend_from_slice(result);
            action(result_buf, *idx);
        }
    }
}

impl<T: Clone> FixedLenRule for MockMutationRule<T> {
    fn match_len(&self) -> usize {
        self.match_len
    }
}

#[derive(Clone, Debug)]
struct ArbReplaceRule<T> {
    remove_idx: usize,
    remove_len: usize,
    replace_with: Vec<T>,
}

impl<T: Arbitrary> Arbitrary for ArbReplaceRule<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        ArbReplaceRule {
            remove_idx: u8::arbitrary(g) as usize,
            remove_len: u8::arbitrary(g) as usize,
            replace_with: Vec::arbitrary(g),
        }
    }
}

fn prep_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<ReplaceRule<MockMutationRule<T>>> {
    let rule = ReplaceRule {
        matcher: MockMutationRule {
            matches: vec![(match_val.clone(), match_idx)],
            matches_at: vec![match_val.clone()],
            match_len: repl_args.remove_len,
        },
        remove_idx: repl_args.remove_idx,
        remove_len: repl_args.remove_len,
        replace_with: repl_args.replace_with.into(),
    };
    if match_idx + rule.get_remove_end() > match_val.len() {
        None
    } else {
        Some(rule)
    }
}

fn collect_matches<M>(matcher: &M, word: &[M::Alph]) -> Vec<(Vec<M::Alph>, usize)>
where
    M: MutationRule,
    M::Alph: Clone,
{
    let mut results: Vec<(Vec<M::Alph>, usize)> = Vec::new();
    matcher.for_each_match(
        word,
        |result, index| results.push((result.clone(), index)),
        &mut Vec::new(),
    );
    results
}

fn collect_matches_at<M>(matcher: &M, index: usize, word: &[M::Alph]) -> Vec<Vec<M::Alph>>
where
    M: MutationRule,
    M::Alph: Clone,
{
    let mut results: Vec<Vec<M::Alph>> = Vec::new();
    matcher.for_each_match_at(
        word,
        index,
        |result| results.push(result.clone()),
        &mut Vec::new(),
    );
    results
}

fn check_match_result_reasonable(
    word: &[FonSet],
    pattern: &[FonSet],
    result_buf: &mut Vec<FonSet>,
    word_idx: usize,
) {
    assert_eq!(result_buf.len(), word.len());
    assert!(word_idx <= word.len() - pattern.len());
    assert_eq!(result_buf[0..word_idx], word[0..word_idx]);
    assert!(!result_buf.is_empty_seq());
}

#[quickcheck]
fn test_fonsetseq_for_each_match_fuzz(word: NonEmptyFonSetSeq, pattern: NonEmptyFonSetSeq) {
    let word = word.as_ref();
    let pattern = pattern.as_ref();
    pattern.for_each_match(
        word,
        |result_buf, word_idx| {
            check_match_result_reasonable(word, pattern, result_buf, word_idx);
        },
        &mut Vec::new(),
    );
}

#[quickcheck]
fn test_fonsetseq_for_each_match_at_fuzz(
    word: WithIndex<NonEmptyFonSetSeq>,
    pattern: NonEmptyFonSetSeq,
) {
    let word_idx = word.index;
    let word = word.item.as_ref();
    let pattern = pattern.as_ref();
    pattern.for_each_match_at(
        word,
        word_idx,
        |result_buf| {
            check_match_result_reasonable(word, pattern, result_buf, word_idx);
        },
        &mut Vec::new(),
    );
}

#[test]
fn test_fon_set_seq_for_each_match_start() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["a"])?, &reg.setseq(&["ax", "bx"])?),
        vec![(reg.setseq(&["a", "bx"])?, 0),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_end() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["b"])?, &reg.setseq(&["ax", "bx"])?),
        vec![(reg.setseq(&["ax", "b"])?, 1),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_middle() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["b"])?, &reg.setseq(&["ax", "bx", "cx"])?),
        vec![(reg.setseq(&["ax", "b", "cx"])?, 1),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_multiple() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(
            &reg.setseq(&["ab", "c"])?,
            &reg.setseq(&["r", "a", "cab", "cx", "na"])?
        ),
        vec![
            (reg.setseq(&["r", "a", "c", "cx", "na"])?, 1),
            (reg.setseq(&["r", "a", "ab", "c", "na"])?, 2),
        ]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_no_match() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["n"])?, &reg.setseq(&["a", "b"])?),
        vec![]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_longer_pattern() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["a", "b"])?),
        vec![]
    );
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["b", "c"])?),
        vec![]
    );
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["a"])?),
        vec![]
    );
    Ok(())
}

fn match_mock_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<Vec<T>> {
    prep_replace_rule(match_val, match_idx, repl_args).map(|rule| {
        let (result, _) = collect_matches(&rule, &[]).into_iter().next().unwrap();
        result
    })
}

fn match_at_mock_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<Vec<T>> {
    prep_replace_rule(match_val, match_idx, repl_args).map(|rule| {
        collect_matches_at(&rule, match_idx, &[])
            .into_iter()
            .next()
            .unwrap()
    })
}

#[quickcheck]
fn replace_rule_for_each_match_inserts(
    match_val: Vec<FonId>,
    match_idx: u8,
    repl_args: ArbReplaceRule<FonId>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_mock_replace_rule(match_val, match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx..].starts_with(&repl_args.replace_with),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_at_inserts(
    match_val: Vec<FonId>,
    match_idx: u8,
    repl_args: ArbReplaceRule<FonId>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_at_mock_replace_rule(match_val, match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx..].starts_with(&repl_args.replace_with),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_removes(
    match_val: Vec<FonId>,
    match_idx: u8,
    repl_args: ArbReplaceRule<FonId>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_mock_replace_rule(match_val.clone(), match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx + repl_args.replace_with.len()..]
                .starts_with(&match_val[match_idx + repl_args.remove_idx + repl_args.remove_len..]),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_at_removes(
    match_val: Vec<FonId>,
    match_idx: u8,
    repl_args: ArbReplaceRule<FonId>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) =
        match_at_mock_replace_rule(match_val.clone(), match_idx, repl_args.clone())
    {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx + repl_args.replace_with.len()..]
                .starts_with(&match_val[match_idx + repl_args.remove_idx + repl_args.remove_len..]),
        )
    } else {
        TestResult::discard()
    }
}

#[test]
fn replace_rule_finding_nothing_new() -> Result<()> {
    let mut reg = FonRegistry::new();
    let word = reg.setseq(&["abc", "def"])?;
    let pattern = reg.setseq(&["ab", "cd"])?;
    let rule = ReplaceRule {
        matcher: pattern,
        remove_idx: 1,
        remove_len: 1,
        replace_with: reg.setseq(&["ef"])?.into(),
    };
    let matches = collect_matches(&rule, &word);
    assert_eq!(matches.len(), 1);
    assert!(matches[0].0.is_subset_of_seq(&word));
    Ok(())
}

#[quickcheck]
fn test_replace_rule_set_for_each_match(matches: Vec<(Vec<FonId>, usize)>) -> bool {
    let rule = MockMutationRule {
        matches,
        ..Default::default()
    };
    let mut rule_set = ReplaceRuleSet::new();
    rule_set.add_any_rule(rule.clone());
    collect_matches(&rule_set, &[]) == collect_matches(&rule, &[])
}

#[quickcheck]
fn test_replace_rule_set_for_each_match_at(matches_at: Vec<Vec<FonId>>) -> bool {
    let rule = MockMutationRule {
        matches_at,
        ..Default::default()
    };
    let mut rule_set = ReplaceRuleSet::new();
    rule_set.add_any_rule(rule.clone());
    collect_matches_at(&rule_set, 0, &[]) == collect_matches_at(&rule, 0, &[])
}

#[test]
fn test_replace_rule_cost_set_add_any() {
    let rule: MockMutationRule<i32> = Default::default();
    let mut rule_set = ReplaceRuleCostSet::new();
    let cost: Cost = 3;
    rule_set.add_any_rule(rule.clone(), cost);
    let idx = rule_set.costs.iter().position(|&c| c == cost).unwrap();
    assert!(rule_set.rules[idx].any_rules.contains(&rule));
}

#[quickcheck]
fn test_start_anchored_fonsetseq_rule(word: NonEmptyFonSetSeq, pattern: NonEmptyFonSetSeq) {
    let word = word.as_ref();
    let pattern = pattern.as_ref();
    let rule = StartAnchoredRule(pattern);
    assert_eq!(
        collect_matches(&rule, word),
        Vec::from_iter(
            collect_matches_at(&pattern, 0, word)
                .into_iter()
                .map(|r| (r, 0))
        )
    );
}

#[quickcheck]
fn test_end_anchored_fonsetseq_rule(word: NonEmptyFonSetSeq, pattern: NonEmptyFonSetSeq) {
    let word = word.as_ref();
    let pattern = pattern.as_ref();
    let rule = EndAnchoredRule(pattern);
    if pattern.len() > word.len() {
        assert_eq!(collect_matches(&rule, word), vec![]);
    } else {
        let index = word.len() - pattern.len();
        assert_eq!(
            collect_matches(&rule, word),
            Vec::from_iter(
                collect_matches_at(&pattern, index, word)
                    .into_iter()
                    .map(|r| (r, index))
            )
        );
    }
}

#[test]
fn test_both_anchored_fonsetseq_rule_same_len() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(
            &BothAnchoredRule(reg.setseq(&["ab", "cd", "ef"])?),
            &reg.setseq(&["b", "c", "e"])?
        ),
        vec![(reg.setseq(&["b", "c", "e"])?, 0)]
    );
    assert_eq!(
        collect_matches(
            &BothAnchoredRule(reg.setseq(&["ab", "cd"])?),
            &reg.setseq(&["a", "b"])?
        ),
        vec![]
    );
    Ok(())
}

#[test]
fn test_both_anchored_fonsetseq_rule_longer_rule() {
    let f = FonSet::from(Fon::from(5u8));
    assert_eq!(collect_matches(&BothAnchoredRule(vec![f, f]), &[f]), vec![]);
    assert_eq!(
        collect_matches(&BothAnchoredRule(vec![f, f, f]), &[f, f]),
        vec![]
    );
}

#[test]
fn test_both_anchored_fonsetseq_rule_longer_word() {
    let f = FonSet::from(Fon::from(5u8));
    assert_eq!(collect_matches(&BothAnchoredRule(vec![f]), &[f, f]), vec![]);
    assert_eq!(
        collect_matches(&BothAnchoredRule(vec![f, f]), &[f, f, f]),
        vec![]
    );
}
