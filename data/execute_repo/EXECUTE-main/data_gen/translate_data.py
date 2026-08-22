from sentence_splitter import SentenceSplitter
from datasets import Dataset
import pycountry
import googletrans
import asyncio
import argparse 
import os
import typing
from googletrans.models import Translated
from types import MethodType

EXCLUDES = ("en", "ca", "fr")

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True, help="Language, full name or lang code.")
parser.add_argument("--resume", action="store_true", help="Resume from temporary file.")

lang2code = googletrans.LANGCODES
GT_LANGS = set(lang2code.keys())
GT_CODES = set(lang2code.values())

def load_prep_data():
    with open("./data/TinyStoriesV2-GPT4-valid.txt", 'r') as file:
        text = file.read().split("<|endoftext|>")[1:] # first story is cut off


    splitter = SentenceSplitter(language="en")
    split = []
    for i in range(len(text)):
        tokens = splitter.split(text[i])
        split.extend(tokens)
    split_ds = Dataset.from_dict({'text': split})
    # filter sentences longer than 10 words, shorter than 3 words
    split_ds = split_ds.filter(lambda x: len(x['text'].split()) <= 10)
    split_ds = split_ds.filter(lambda x: len(x['text'].split()) >= 3)
    # filter out sentences that don't end with a .!? (these are mostly garbage like titles)
    split_ds = split_ds.filter(lambda x: x['text'][-1] in ['.','!','?'])
    # filter out sentences with "
    split_ds = split_ds.filter(lambda x: '"' not in x['text'])
    # shuffle
    split_ds = split_ds.shuffle(seed=0)
    # cut dataset to 5k examples
    split_ds = split_ds.select(range(5000))
    return split_ds

async def translate(examples, lang_code, resume):
    tgt_text = {'text': []}
    done = 0

    os.makedirs(f'./data/translated/temp', exist_ok=True)
    temp_file = open(f'./data/translated/temp/{lang_code}.txt', 'a')
    temp_file_num_lines = sum(1 for _ in open(f'./data/translated/temp/{lang_code}.txt'))

    async with googletrans.Translator() as translator:
        translator.translate = MethodType(gtranslate, translator)
        for i, src_text in enumerate(examples['text']):
            if resume and i < temp_file_num_lines:
                continue
                
            translation = await translator.translate(src_text, src='en', dest=lang_code)
            tgt_text['text'].append(translation.text)
            done += 1
            if done % 10 == 0 and done > 0:
                print(i, tgt_text['text'][-1], flush=True)

            temp_file.write(translation.text + '\n')

    temp_file.close()
    with open(f'./data/translated/temp/{lang_code}.txt', 'r') as f:
        tgt_text['text'] = f.read().split('\n')

    return tgt_text

def print_dataset(ds1, ds2, filename):
    with open(filename, 'w') as f:
        for i in range(len(ds1)):
            f.write(ds1[i]['text'] + '\t' + ds2[i]['text'] + '\n')

def get_code(lang):
    if lang in ["iu", "sat"]:
        return lang
    elif lang in GT_LANGS: # using language name
        code = lang2code[lang]
    elif lang in GT_CODES: # using ISO code
        code = lang
    else: # using the wrong ISO code
        if len(lang) == 3: # if I provide the ISO-3 but google needs ISO-2
            code = pycountry.languages.get(alpha_3=lang).alpha_2
        elif len(lang) == 2: # if I provide the ISO-2 but google needs ISO-3
            code = pycountry.languages.get(alpha_2=lang).alpha_3

        if code not in GT_CODES:
            raise ValueError(f"Language {lang} not supported by Google Translate.")
        
    return code



def main():
    args = parser.parse_args()
    if "," in args.lang:
        langs = args.lang.split(",")
        codes = [get_code(lang.strip()) for lang in langs]
    else:
        codes = [get_code(args.lang.strip())]
    
    iso3_codes = [pycountry.languages.get(alpha_2=code).alpha_3 if len(code) == 2 else code for code in codes]

    dataset = load_prep_data()
    os.makedirs(f"./data/translated", exist_ok=True)

    for iso3_code, code in zip(iso3_codes, codes):
        if iso3_code == "eng": # already an english dataset
            xdataset = dataset
        else:
            results = asyncio.run(translate(dataset, code, args.resume))
            xdataset = Dataset.from_dict(results)

        # save
        xdataset.save_to_disk(f"./data/translated/{iso3_code}")

# override to allow languages like santali
async def gtranslate(
    self,
    text: typing.Union[str, typing.List[str]],
    dest: str = "en",
    src: str = "auto",
    **kwargs: typing.Any,
) -> typing.Union[Translated, typing.List[Translated]]:
    """Translate text from source language to destination language

    :param text: The source text(s) to be translated. Batch translation is supported via sequence input.
    :type text: UTF-8 :class:`str`; :class:`unicode`; string sequence (list, tuple, iterator, generator)

    :param dest: The language to translate the source text into.
                    The value should be one of the language codes listed in :const:`googletrans.LANGUAGES`
                    or one of the language names listed in :const:`googletrans.LANGCODES`.
    :param dest: :class:`str`; :class:`unicode`

    :param src: The language of the source text.
                The value should be one of the language codes listed in :const:`googletrans.LANGUAGES`
                or one of the language names listed in :const:`googletrans.LANGCODES`.
                If a language is not specified,
                the system will attempt to identify the source language automatically.
    :param src: :class:`str`; :class:`unicode`

    :rtype: Translated
    :rtype: :class:`list` (when a list is passed)

    Basic usage:
        >>> from googletrans import Translator
        >>> translator = Translator()
        >>> translator.translate('안녕하세요.')
        <Translated src=ko dest=en text=Good evening. pronunciation=Good evening.>
        >>> translator.translate('안녕하세요.', dest='ja')
        <Translated src=ko dest=ja text=こんにちは。 pronunciation=Kon'nichiwa.>
        >>> translator.translate('veritas lux mea', src='la')
        <Translated src=la dest=en text=The truth is my light pronunciation=The truth is my light>

    Advanced usage:
        >>> translations = translator.translate(['The quick brown fox', 'jumps over', 'the lazy dog'], dest='ko')
        >>> for translation in translations:
        ...    print(translation.origin, ' -> ', translation.text)
        The quick brown fox  ->  빠른 갈색 여우
        jumps over  ->  이상 점프
        the lazy dog  ->  게으른 개
    """
    dest = dest.lower().split("_", 1)[0]
    src = src.lower().split("_", 1)[0]

    if isinstance(text, list):
        concurrency_limit = kwargs.pop(
            "list_operation_max_concurrency", self.list_operation_max_concurrency
        )
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def translate_with_semaphore(item):
            async with semaphore:
                return await self.translate(item, dest=dest, src=src, **kwargs)

        tasks = [translate_with_semaphore(item) for item in text]
        result = await asyncio.gather(*tasks)
        return result

    origin = text
    data, response = await self._translate(text, dest, src, kwargs)

    # this code will be updated when the format is changed.
    translated = "".join([d[0] if d[0] else "" for d in data[0]])

    extra_data = self._parse_extra_data(data)

    # actual source language that will be recognized by Google Translator when the
    # src passed is equal to auto.
    try:
        src = data[2]
    except Exception:  # pragma: nocover
        pass

    pron = origin
    try:
        pron = data[0][1][-2]
    except Exception:  # pragma: nocover
        pass

    if pron is None:
        try:
            pron = data[0][1][2]
        except:  # pragma: nocover  # noqa: E722
            pass

    if dest in EXCLUDES and pron == origin:
        pron = translated

    # put final values into a new Translated object
    result = Translated(
        src=src,
        dest=dest,
        origin=origin,
        text=translated,
        pronunciation=pron,
        extra_data=extra_data,
        response=response,
    )

    return result


if __name__ == "__main__":
    main()