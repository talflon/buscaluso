# Buscaluso

Searching for sound-alike and etymologically-related words in a dictionary.

Built to be used with a Portuguese dictionary and words as heard from around the Lusophone world (hence the name), especially [Cape Verdean Creole](https://en.wikipedia.org/wiki/Cape_Verdean_Creole) from [São Vicente](https://en.wikipedia.org/wiki/S%C3%A3o_Vicente,_Cape_Verde). Intended to be a tool to help answer questions like the following:

- Is there a Portuguese word with similar meaning and etymology to "txi"? *(descer)*
- Could "tcheu" meaning *muito* have come from Portuguese? *(cheio)*
- Sounds like "quat", but context seems to be art? *(quadro)*
- How do you spell "ilya"? *(ilha)*
- What did that speaker from Lisbon say that sounds like "bain"? *(bem)*
- What did that speaker from northern Portugal say that sounds like "baka"? *(vaca)*
- What did that speaker from Brazil say that sounds like "tchia"? *(tia)*

This is currently under development, and not in good shape yet. Eventual goals:
- better performance
  - finding more appropriate words
  - finding words quicker
  - keeping resource usage under control
- web API

Non-goals:
- linguistic correctness

## Usage

To start a command-line search words from a particular starting point `[WORD]`:

```
buscaluso [OPTIONS] --rules <RULES> --dict <DICT> [WORD]
```

`<DICT>` should be a dictionary of words, one word per line. On a Linux distribution, there should be a package you can install which will put such a file into `/usr/share/dict/`. (On Debian the package is `wportuguese`.)

`<RULES>` is a file defining the rules to search by. See [Rule file format](#rule-file-format) for details.

Currently, the program will just keep trying to find more words until your computer runs out of memory.

If you don't specify any `[WORD]`, it will load all the rules and enter an interactive mode.
In interactive mode, you can type a word to start a new search for that word,
or just hit enter to look for the next word on the current search.

## How it works

Buscaluso normalizes the word you input, and the words from its dictionary, into a different alphabet.
I've been calling the letters of this alphabet **fons**, because "fon" sounds a little like "phoneme", but is not spelled the same as anything else.
They're intended to be similar to, but not identical to, a phonological representation.
Basically: whatever's most convenient to define the rules for word similarity.

After buscaluso normalizes the word into a sequence of fons,
it applies mutation rules to them in a breadth-first search,
and returns matches from its dictionary.

## Rule file format

A rule file is a UTF-8 text file, with one rule per line.
Comments start with a semicolon (`;`) and last until the end of the line.
*Fons* are named as sequences of lowercase letters, unicode allowed.
Most are single letters, but since they can have multiple letters, they must be separated from each other by whitespace.
There are three types of rules:

### Alias rules

These define aliases for sets of *fons*. The alias name starts with a capital letter. For example:

```
V = [a e i o u]  ; vowels
```

### Normalization rules

These define the transformation from sequences of letters (on the left of the `>`) in the input and dictionary words
to the internal *fon* representation (on the right of the `>`).
Rules are applied as longest-rule first, moving left to right.
If no rules apply, and the letter being looked at is used as a fon somewhere in the rules file, then it's used as is. For example:

```
s s > ç
c i > ç i
c e > ç e
```

Under these rules, "ç", "ss", and "c" before "i" or "e" are represented as "ç".

An underscore (`_`) means "nothing", and can be used for both anchoring the rule to the start or end of the word, or to indicate an empty output.

Sets of letters can be specified on the left, surrounded by `[ ]` and separated by spaces, but not on the right.

### Mutation rules

These define the possible steps in the search for matching words, by changing the *fon* representations around.

An example of a full rule with all its elements:

```
20: C | i u > _ | _
```

The initial number is the "cost" of the rule. Cheaper rules are tried first.

The pipe symbols (`|`) separate look-ahead and look-behind sections. These are not changed by the rule.

In the middle is a rule that looks similar to the normalization rules, but it's from fon representation to fon representation.

An underscore (`_`) means "nothing", and can be used for both anchoring the rule to the start or end of the word, or to indicate an empty output on either side of the `>`.

Rules are applied in both directions: There's no need to define

```
20: C | _ > i u | _
```

because the first definition defines both.