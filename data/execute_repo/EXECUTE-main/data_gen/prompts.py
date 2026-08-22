PROMPTS = {
    "spell": 'Spell out the word, putting spaces between each letter, based on the following examples:\n\
            \n\
            1. Spell out the word " {} ". Answer: " {} "\n\
            2. Spell out the word " {} ". Answer: " {} "\n\
            3. Spell out the word " {} ". Answer: " {} "\n\
            4. Spell out the word " {} ". Answer: " {} "\n\
            \n\
            Question: Spell out the word " {} ".',
    "spell_inverse": 'Write the word that is spelled out, without any spaces, based on the following examples:\n\
            \n\
            1. Write the word " {} ".  Answer: " {} "\n\
            2. Write the word " {} ". Answer: " {} "\n\
            3. Write the word " {} ". Answer: " {} "\n\
            4. Write the word " {} ". Answer: " {} "\n\
            \n\
            Question: Write the word " {} ".',
    "contains_char": 'Answer whether the specified letter is in the given word, based on the following examples:\n\
            \n\
            1. Is there a " {} " in " {} "? Answer: " {} "\n\
            2. Is there a " {} " in " {} "? Answer: " {} "\n\
            3. Is there a " {} " in " {} "? Answer: " {} "\n\
            4. Is there a " {} " in " {} "? Answer: " {} "\n\
            \n\
            Question: Is there a " {} " in " {} "?',
    "contains_word": 'Answer whether the specified word is in the given sentence (case insensitive), based on the following examples:\n\
            \n\
            1. Is there a " {} " in " {} "? Answer: " {} "\n\
            2. Is there a " {} " in " {} "? Answer: " {} "\n\
            3. Is there a " {} " in " {} "? Answer: " {} "\n\
            4. Is there a " {} " in " {} "? Answer: " {} "\n\
            \n\
            Question: Is there a " {} " in " {} "?',
    "ins_char": 'Add the specified letter after every instance of the second specified letter in a given word, based on the following examples:\n\
            \n\
            1. Add an " {} " after every " {} " in " {} ". Answer: " {} "\n\
            2. Add an " {} " after every " {} " in " {} ". Answer: " {} "\n\
            3. Add an " {} " after every " {} " in " {} ". Answer: " {} "\n\
            4. Add an " {} " after every " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Add an " {} " after every " {} " in " {} ".',
    "ins_word": 'Add the specified word after every instance of the second specified word in a given sentence, based on the following examples:\n\
            \n\
            1. Add " {} " after every " {} " in " {} ". Answer: " {} "\n\
            2. Add " {} " after every " {} " in " {} ". Answer: " {} "\n\
            3. Add " {} " after every " {} " in " {} ". Answer: " {} "\n\
            4. Add " {} " after every " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Add " {} " after every " {} " in " {} ".',
    "del_char": 'Delete every instance of a specified letter in a given word, based on the following examples:\n\
            \n\
            1. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            2. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            3. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            4. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Delete every instance of " {} " in " {} ".',
    "del_word": 'Delete every instance of a specified word in a given sentence, based on the following examples:\n\
            \n\
            1. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            2. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            3. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            4. Delete every instance of " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Delete every instance of " {} " in " {} ".',
    "swap_char": 'Swap the positions of two specified letters in a given word, based on the following examples:\n\
            \n\
            1. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            2. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            3. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            4. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Swap " {} " and " {} " in " {} ".',
    "swap_word": 'Swap the positions of two specified words in a given sentence, based on the following examples:\n\
            \n\
            1. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            2. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            3. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            4. Swap " {} " and " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Swap " {} " and " {} " in " {} ".',
    "sub_char": 'Substitute the first specified letter with the second specified letter in a given word, based on the following examples:\n\
            \n\
            1. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            2. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            3. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            4. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Substitute " {} " with " {} " in " {} ".',
    "sub_word": 'Substitute the first specified word with the second specified word in a given sentence, based on the following examples:\n\
            \n\
            1. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            2. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            3. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            4. Substitute " {} " with " {} " in " {} ". Answer: " {} "\n\
            \n\
            Question: Substitute " {} " with " {} " in " {} ".',
}

EXTRA_PROMPTS = {
    "kor_g2c": 'Write out the Hangul character in individual Jamo, based on the following examples:\n\
            \n\
            1. Write out the Hangul character " {} ". Answer: " {} "\n\
            2. Write out the Hangul character " {} ". Answer: " {} "\n\
            3. Write out the Hangul character " {} ". Answer: " {} "\n\
            4. Write out the Hangul character " {} ". Answer: " {} "\n\
            \n\
            Question: Write out the Hangul character " {} ".',
    "kor_c2g": 'Combine the Jamo characters into Hangul, based on the following examples:\n\
            \n\
            1. Combine the Jamo characters " {} ". Answer: " {} "\n\
            2. Combine the Jamo characters " {} ". Answer: " {} "\n\
            3. Combine the Jamo characters " {} ". Answer: " {} "\n\
            4. Combine the Jamo characters " {} ". Answer: " {} "\n\
            \n\
            Question: Combine the Jamo characters " {} ".',
    "jpn_g2c": 'Split the character into its Kanji radicals, based on the following examples:\n\
            \n\
            1. Split the character " {} " into its Kanji radicals. Answer: " {} "\n\
            2. Split the character " {} " into its Kanji radicals. Answer: " {} "\n\
            3. Split the character " {} " into its Kanji radicals. Answer: " {} "\n\
            4. Split the character " {} " into its Kanji radicals. Answer: " {} "\n\
            \n\
            Question: Split the character " {} " into its Kanji radicals.',
    "jpn_c2g": 'Combine the Kanji radicals into a character, based on the following examples:\n\
            \n\
            1. Combine the Kanji radicals " {} " into a character. Answer: " {} "\n\
            2. Combine the Kanji radicals " {} " into a character. Answer: " {} "\n\
            3. Combine the Kanji radicals " {} " into a character. Answer: " {} "\n\
            4. Combine the Kanji radicals " {} " into a character. Answer: " {} "\n\
            \n\
            Question: Combine the Kanji radicals " {} " into a character.',
    "zho_g2c": 'Split the character into its Kangxi radicals, based on the following examples:\n\
            \n\
            1. Split the character " {} " into its Kangxi radicals. Answer: " {} "\n\
            2. Split the character " {} " into its Kangxi radicals. Answer: " {} "\n\
            3. Split the character " {} " into its Kangxi radicals. Answer: " {} "\n\
            4. Split the character " {} " into its Kangxi radicals. Answer: " {} "\n\
            \n\
            Question: Split the character " {} " into its Kangxi radicals.',
    "zho_c2g": 'Combine the Kangxi radicals into a character, based on the following examples:\n\
            \n\
            1. Combine the Kangxi radicals " {} " into a character. Answer: " {} "\n\
            2. Combine the Kangxi radicals " {} " into a character. Answer: " {} "\n\
            3. Combine the Kangxi radicals " {} " into a character. Answer: " {} "\n\
            4. Combine the Kangxi radicals " {} " into a character. Answer: " {} "\n\
            \n\
            Question: Combine the Kangxi radicals " {} " into a character.',
    "jpn_k2h": 'Translate the Kanji character into Hiragana, based on the following examples:\n\
            \n\
            1. Translate the Kanji character " {} " into Hiragana. Answer: " {} "\n\
            2. Translate the Kanji character " {} " into Hiragana. Answer: " {} "\n\
            3. Translate the Kanji character " {} " into Hiragana. Answer: " {} "\n\
            4. Translate the Kanji character " {} " into Hiragana. Answer: " {} "\n\
            \n\
            Question: Translate the Kanji character " {} " into Hiragana.',
    "jpn_h2k": 'Translate the Hiragana characters into Kanji, based on the following examples:\n\
            \n\
            1. Translate the Hiragana characters " {} " into Kanji. Answer: " {} "\n\
            2. Translate the Hiragana characters " {} " into Kanji. Answer: " {} "\n\
            3. Translate the Hiragana characters " {} " into Kanji. Answer: " {} "\n\
            4. Translate the Hiragana characters " {} " into Kanji. Answer: " {} "\n\
            \n\
            Question: Translate the Hiragana characters " {} " into Kanji.',
    "zho_contains": 'Answer whether the specified Kangxi radical is in the given character, based on the following examples:\n\
            \n\
            1. Is there a " {} " in " {} "? Answer: " {} "\n\
            2. Is there a " {} " in " {} "? Answer: " {} "\n\
            3. Is there a " {} " in " {} "? Answer: " {} "\n\
            4. Is there a " {} " in " {} "? Answer: " {} "\n\
            \n\
            Question: Is there a " {} " in " {} "?',
    "jpn_contains": 'Answer whether the specified Kanji radical is in the given character, based on the following examples:\n\
            \n\
            1. Is there a " {} " in " {} "? Answer: " {} "\n\
            2. Is there a " {} " in " {} "? Answer: " {} "\n\
            3. Is there a " {} " in " {} "? Answer: " {} "\n\
            4. Is there a " {} " in " {} "? Answer: " {} "\n\
            \n\
            Question: Is there a " {} " in " {} "?',
    "kor_contains": 'Answer whether the specified Jamo is in the given Hangul character, based on the following examples:\n\
            \n\
            1. Is there a " {} " in " {} "? Answer: " {} "\n\
            2. Is there a " {} " in " {} "? Answer: " {} "\n\
            3. Is there a " {} " in " {} "? Answer: " {} "\n\
            4. Is there a " {} " in " {} "? Answer: " {} "\n\
            \n\
            Question: Is there a " {} " in " {} "?',
}