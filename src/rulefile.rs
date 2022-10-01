use super::*;

use core::fmt;

use nom::branch::alt;
use nom::bytes::complete::take_while1;
use nom::character::complete::{anychar, char, digit1, space0, space1};
use nom::combinator::{eof, map, map_res, opt, verify};
use nom::multi::separated_list1;
use nom::sequence::{delimited, preceded, separated_pair, terminated, tuple};
use nom::IResult;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule<'a> {
    Alias(&'a str, ItemSet<'a>),
    Norm {
        from: ItemSetSeq<'a>,
        to: ItemSeq<'a>,
    },
    Mut {
        cost: Cost,
        before: ItemSetSeq<'a>,
        from: ItemSetSeq<'a>,
        to: ItemSetSeq<'a>,
        after: ItemSetSeq<'a>,
    },
}

type IRes<'a, T> = IResult<&'a str, T>;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Item<'a> {
    Char(char),
    Alias(&'a str),
    None,
}

// TODO: read https://github.com/apolitical/impl-display-for-vec and possible apply here

pub type ItemSet<'a> = Vec<Item<'a>>;

pub type ItemSeq<'a> = Vec<Item<'a>>;

pub type ItemSetSeq<'a> = Vec<ItemSet<'a>>;

impl<'a> fmt::Display for Item<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Item::Char(c) => write!(f, "{}", c),
            Item::Alias(name) => write!(f, "{}", name),
            Item::None => write!(f, "_"),
        }
    }
}

pub fn item_set_seq_to_str(seq: &ItemSetSeq) -> String {
    seq.iter()
        .map(item_set_to_str)
        .reduce(|a, b| format!("{} {}", a, b))
        .unwrap_or_default()
}

pub fn item_set_to_str(seq: &ItemSet) -> String {
    format!("[{}]", item_seq_to_str(seq))
}

pub fn item_seq_to_str(seq: &ItemSeq) -> String {
    seq.iter()
        .map(|item| format!("{}", item))
        .reduce(|a, b| format!("{} {}", a, b))
        .unwrap_or_default()
}

fn alias_name(input: &str) -> IRes<&str> {
    verify(take_while1(|c: char| c.is_alphanumeric()), |s: &str| {
        s.chars().next().unwrap().is_uppercase()
    })(input)
}

fn is_valid_char(c: char) -> bool {
    c.is_lowercase() || c == '-' || c == '\''
}

fn item(input: &str) -> IRes<Item> {
    alt((
        map(char('_'), |_| Item::None),
        map(verify(anychar, |&c| is_valid_char(c)), Item::Char),
        map(alias_name, Item::Alias),
    ))(input)
}

fn item_seq(input: &str) -> IRes<ItemSeq> {
    separated_list1(space1, item)(input)
}

fn item_set(input: &str) -> IRes<ItemSet> {
    alt((
        delimited(char('['), separated_list1(space1, item), char(']')),
        map(item, |i| vec![i]),
    ))(input)
}

fn item_set_seq(input: &str) -> IRes<ItemSetSeq> {
    separated_list1(space1, item_set)(input)
}

fn cost(input: &str) -> IRes<Cost> {
    map_res(digit1, |s| Cost::from_str_radix(s, 10))(input)
}

fn alias_rule(input: &str) -> IRes<Rule> {
    map(
        separated_pair(alias_name, delimited(space0, char('='), space0), item_set),
        |(name, value)| Rule::Alias(name, value),
    )(input)
}

fn norm_rule(input: &str) -> IRes<Rule> {
    map(
        separated_pair(item_set_seq, delimited(space0, char('>'), space0), item_seq),
        |(before, after)| Rule::Norm {
            from: before,
            to: after,
        },
    )(input)
}

fn mut_rule(input: &str) -> IRes<Rule> {
    map(
        tuple((
            cost,
            delimited(space0, char(':'), space0),
            opt(terminated(
                item_set_seq,
                delimited(space0, char('|'), space0),
            )),
            item_set_seq,
            delimited(space0, char('>'), space0),
            item_set_seq,
            opt(preceded(delimited(space0, char('|'), space0), item_set_seq)),
        )),
        |(cost, _, look_behind, before, _, after, look_ahead)| Rule::Mut {
            cost,
            before: look_behind.unwrap_or_default(),
            from: before,
            to: after,
            after: look_ahead.unwrap_or_default(),
        },
    )(input)
}

fn rule(input: &str) -> IRes<Rule> {
    alt((alias_rule, norm_rule, mut_rule))(input)
}

fn remainder(input: &str) -> IRes<&str> {
    Ok(("", input))
}

fn comment(input: &str) -> IRes<&str> {
    preceded(char(';'), remainder)(input)
}

pub fn rule_line(input: &str) -> IRes<Option<Rule>> {
    terminated(
        delimited(space0, opt(rule), preceded(space0, opt(comment))),
        eof,
    )(input)
}
