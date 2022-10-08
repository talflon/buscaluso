// Copyright © 2022 Daniel Getz
// SPDX-License-Identifier: MIT

use super::fon::tests::*;
use super::fon::*;
use super::*;

use std::collections::BTreeSet;

use ntest::timeout;
use quickcheck::{Arbitrary, QuickCheck, TestResult};
use quickcheck_macros::*;

use rulefile::Item;

#[derive(Debug, Clone)]
pub struct NonEmptyVec<T>(Vec<T>);

impl<T: Arbitrary> Arbitrary for NonEmptyVec<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        loop {
            let v = Vec::arbitrary(g);
            if !v.is_empty() {
                return NonEmptyVec(v);
            }
        }
    }
}

impl<T> AsRef<[T]> for NonEmptyVec<T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl<T> IntoIterator for NonEmptyVec<T> {
    type Item = T;

    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct WithIndex<A: Arbitrary> {
    pub item: A,
    pub index: usize,
}

impl<A> Arbitrary for WithIndex<A>
where
    A: Arbitrary,
    A: IntoIterator,
    A: AsRef<[A::Item]>,
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        loop {
            let item = A::arbitrary(g);
            let len = item.as_ref().len();
            if len > 0 {
                let index = usize::arbitrary(g) % len;
                return WithIndex { item, index };
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct WithInsIndex<A: Arbitrary> {
    pub item: A,
    pub index: usize,
}

impl<A> Arbitrary for WithInsIndex<A>
where
    A: Arbitrary,
    A: IntoIterator,
    A: AsRef<[A::Item]>,
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let item = A::arbitrary(g);
        let index = usize::arbitrary(g) % (item.as_ref().len() + 1);
        WithInsIndex { item, index }
    }
}

fn collect_words_normalized<'a>(dict: &'a BTreeDictionary, normalized: &[Fon]) -> Vec<&'a str> {
    let mut results = Vec::new();
    dict.for_each_word_normalized(normalized, |s| results.push(s));
    results
}

fn sorted<T: Ord>(mut vec: Vec<T>) -> Vec<T> {
    vec.sort();
    vec
}

#[test]
fn test_dictionary_one_key() {
    let normalized = &[Fon::from(8u8), Fon::from(2u8)];
    let mut dict = BTreeDictionary::new();
    assert_eq!(collect_words_normalized(&dict, normalized), &[] as &[&str]);

    dict.add_word(normalized, "first");
    assert_eq!(collect_words_normalized(&dict, normalized), &["first"]);

    dict.add_word(normalized, "another");
    assert_eq!(
        sorted(collect_words_normalized(&dict, normalized)),
        &["another", "first"]
    );
}

#[test]
fn test_dictionary_two_keys() {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(41u8)];
    let mut dict = BTreeDictionary::new();
    dict.add_word(norm1, "first");
    dict.add_word(norm1, "another");
    assert_eq!(collect_words_normalized(&dict, norm2), &[] as &[&str]);

    dict.add_word(norm2, "word");
    assert_eq!(collect_words_normalized(&dict, norm2), &["word"]);

    assert_eq!(
        sorted(collect_words_normalized(&dict, norm1)),
        &["another", "first"]
    );
}

#[test]
fn test_dictionary_duplicate_one_key() {
    let key = &[Fon::from(33u8)];
    let mut dict = BTreeDictionary::new();
    assert_eq!(collect_words_normalized(&dict, key), &[] as &[&str]);

    dict.add_word(key, "value");
    assert_eq!(collect_words_normalized(&dict, key), &["value"]);
    dict.add_word(key, "value");
    assert_eq!(collect_words_normalized(&dict, key), &["value"]);
}

#[test]
fn test_dictionary_duplicate_two_keys() {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(41u8)];
    let mut dict = BTreeDictionary::new();
    dict.add_word(norm1, "same");
    dict.add_word(norm2, "same");
    assert_eq!(collect_words_normalized(&dict, norm1), &["same"]);
    assert_eq!(collect_words_normalized(&dict, norm2), &["same"]);
}

#[test]
fn test_buscacfg_normalize() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    for c in "mno".chars() {
        cfg.fon_registry.add(c)?;
    }
    cfg.normalize_rules
        .add_rule(&['a', 'b', 'c'], &[cfg.fon_registry.add('z')?])?;
    assert_eq!(
        cfg.normalize("noabcm")?,
        cfg.fon_registry.normalize("nozm")?
    );
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_buscacfg_search_word_in_dictionary_comes_first() -> Result<()> {
    let word = "word";
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("10: x > y".as_bytes())?;
    for c in word.chars() {
        cfg.fon_registry.add(c)?;
    }
    cfg.dictionary.add_word(&cfg.normalize(word)?, word);
    assert_eq!(cfg.search(word)?.iter().flatten().next(), Some((word, 0)));
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_buscacfg_search_normalize_error() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("10: x > y".as_bytes())?;
    assert!(matches!(cfg.search("anything"), Err(NoSuchFon(_))));
    Ok(())
}

#[quickcheck]
fn test_set_alias(fons: FonSet) -> Result<()> {
    let mut cfg = BuscaCfg::new();
    assert_eq!(cfg.try_get_alias("blah"), None);
    assert!(matches!(cfg.get_alias("blah"), Err(NoSuchAlias(_))));
    cfg.set_alias("blah".into(), fons);
    assert_eq!(cfg.try_get_alias("blah"), Some(fons));
    assert_eq!(cfg.get_alias("blah")?, fons);
    assert_eq!(cfg.try_get_alias("blah!"), None);
    assert!(matches!(cfg.get_alias("blah!"), Err(NoSuchAlias(_))));
    Ok(())
}

#[test]
fn test_add_rule_alias() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Alias("Hi", vec![Item::Char('a')]))?;
    assert_eq!(
        cfg.get_alias("Hi")?,
        FonSet::from(cfg.fon_registry.get_fon('a')?)
    );
    Ok(())
}

#[test]
fn test_add_rule_alias_with_anchor() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Alias("Xy", vec![Item::Char('n'), Item::None]))?;
    assert_eq!(
        cfg.get_alias("Xy")?,
        FonSet::from([cfg.fon_registry.get_fon('n')?, NO_FON])
    );
    Ok(())
}

#[test]
fn test_resolve_norm_rule_result() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let r = cfg.resolve_norm_rule_result(&vec![Item::Char('o'), Item::Char('k')])?;
    assert_eq!(r, cfg.normalize("ok")?);
    Ok(())
}

#[test]
fn test_invalid_normalize_rule_rhs() {
    assert!(matches!(
        BuscaCfg::new().load_rules("a b > a _".as_bytes()),
        Err(InvalidNormRule(_))
    ));
    assert!(matches!(
        BuscaCfg::new().load_rules("a b > _ c".as_bytes()),
        Err(InvalidNormRule(_))
    ));
    assert!(matches!(
        BuscaCfg::new().load_rules("C = [z r]\nv > C".as_bytes()),
        Err(InvalidNormRule(_))
    ));
}

#[test]
fn test_invalid_normalize_rule_lhs() {
    assert!(BuscaCfg::new().load_rules(" > a".as_bytes()).is_err());
    assert!(BuscaCfg::new().load_rules("a _ b > c".as_bytes()).is_err());
    assert!(BuscaCfg::new().load_rules("_ _ _ > c".as_bytes()).is_err());
}

#[test]
fn test_resolve_rule_item_set_alias() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Alias("N")])?;
    assert_eq!(r, s);
    Ok(())
}

#[test]
fn test_resolve_rule_item_set_alias_plus() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Char('q'), Item::Alias("N")])?;
    assert_eq!(r, s | cfg.fon_registry.get_fon('q')?);
    Ok(())
}

#[test]
fn test_resolve_rule_item_set_alias_plus_anchor() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Alias("N"), Item::None])?;
    assert_eq!(r, s | NO_FON);
    Ok(())
}

#[test]
fn test_add_rule_norm_simple() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('a')], vec![Item::Char('b')]],
        to: vec![Item::Char('c'), Item::Char('d')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['a', 'b']),
        Some(cfg.normalize("cd")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_anchored_start() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::None], vec![Item::Char('x')]],
        to: vec![Item::Char('y')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&[NO_FON_CHAR, 'x']),
        Some(cfg.normalize("y")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_anchored_end() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('e')], vec![Item::None]],
        to: vec![Item::Char('r')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['e', NO_FON_CHAR]),
        Some(cfg.normalize("r")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_with_sets() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![
            vec![Item::Char('n')],
            vec![Item::Char('a'), Item::Char('o')],
        ],
        to: vec![Item::Char('y')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['n', 'o']),
        Some(cfg.normalize("y")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_adds_rhs_only_to_registry() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('a')], vec![Item::Char('b')]],
        to: vec![Item::Char('c')],
    })?;
    assert!(cfg.fon_registry.get_fon('c').is_ok());
    assert!(cfg.fon_registry.get_fon('a').is_err());
    assert!(cfg.fon_registry.get_fon('b').is_err());
    Ok(())
}

fn for_cartesian_product_results<T, S>(items: &[S]) -> Result<Vec<Vec<T>>>
where
    S: AsRef<[T]>,
    T: Copy + Default,
{
    let mut results = Vec::new();
    for_cartesian_product(items, |i| {
        results.push(Vec::from(i));
        Ok(())
    })?;
    Ok(results)
}

#[test]
fn test_for_cartesian_product() -> Result<()> {
    assert_eq!(
        for_cartesian_product_results(&[vec![3, 4], vec![7], vec![0, 10, 100]])?,
        vec![
            vec![3, 7, 0],
            vec![3, 7, 10],
            vec![3, 7, 100],
            vec![4, 7, 0],
            vec![4, 7, 10],
            vec![4, 7, 100],
        ]
    );
    Ok(())
}

#[test]
fn test_for_cartesian_product_len() {
    fn for_cartesian_product_len_is_product(items: Vec<Vec<FonId>>) -> TestResult {
        let expected_len: usize = items.iter().map(Vec::len).product();
        if items.is_empty() || expected_len == 0 || expected_len > 10_000 {
            TestResult::discard()
        } else {
            let mut count: usize = 0;
            for_cartesian_product(&items, |_| {
                count += 1;
                Ok(())
            })
            .unwrap();
            TestResult::from_bool(count == expected_len)
        }
    }
    QuickCheck::new()
        .gen(quickcheck::Gen::new(8))
        .quickcheck(for_cartesian_product_len_is_product as fn(_) -> TestResult);
}

#[test]
fn test_for_cartesian_product_contains_samples() {
    fn for_cartesian_product_contains_samples(items: Vec<Vec<FonId>>) -> TestResult {
        let expected_len: usize = items.iter().map(Vec::len).product();
        if items.is_empty() || expected_len == 0 || expected_len > 10_000 {
            TestResult::discard()
        } else {
            let mut looking_for = BTreeSet::new();
            looking_for.insert(Vec::from_iter(items.iter().map(|v| v[0])));
            looking_for.insert(Vec::from_iter(
                items.iter().map(|v| v.last().cloned().unwrap()),
            ));
            looking_for.insert(Vec::from_iter(
                items.iter().map(|v| v.iter().min().cloned().unwrap()),
            ));
            for_cartesian_product(&items, |x| {
                looking_for.remove(x);
                Ok(())
            })
            .unwrap();
            TestResult::from_bool(looking_for.is_empty())
        }
    }
    QuickCheck::new()
        .gen(quickcheck::Gen::new(8))
        .quickcheck(for_cartesian_product_contains_samples as fn(_) -> TestResult);
}

#[test]
fn test_sliceset_with_fon_set_seq() -> Result<()> {
    let mut set: BTreeSet<Box<[FonSet]>> = BTreeSet::new();
    let mut reg = FonRegistry::new();
    let setseq1 = reg.setseq(&["xy", "z"])?;
    let setseq2 = reg.setseq(&["z", "ay"])?;
    assert!(!set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq2);
    assert!(set.has_slice(&setseq1));
    assert!(set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(set.has_slice(&setseq2));
    Ok(())
}

#[test]
fn test_busca_node_inc_cost() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("0: a > b".as_bytes())?;
    cfg.load_rules("12: a > b".as_bytes())?;
    cfg.load_rules("3: a > b".as_bytes())?;
    cfg.load_rules("55: a > b".as_bytes())?;
    let word = cfg.fon_registry.setseq(&["a"])?;
    let mut node = BuscaNode::new(&cfg, &word, 0);
    assert_eq!(node.total_cost, 0);
    assert_eq!(node.inc_cost(&cfg), Some(3));
    assert_eq!(node.inc_cost(&cfg), Some(12));
    assert_eq!(node.inc_cost(&cfg), Some(55));
    assert_eq!(node.inc_cost(&cfg), None);
    Ok(())
}

#[test]
fn test_busca_node_cmp() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("0: a > b".as_bytes())?;
    let word = cfg.fon_registry.setseq(&["a"])?;
    let node1 = BuscaNode::new(&cfg, &word, 0);
    let node2 = BuscaNode::new(&cfg, &word, 78);
    assert!(node1 > node2);
    assert!(node2 < node1);
    assert!(node1 >= node1);
    assert!(node1 <= node1);
    assert!(node2 >= node2);
    assert!(node2 <= node2);
    Ok(())
}
