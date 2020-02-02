import string

def tokenize(main_string, delimeters={" "},
                                      special_tokens={"\\newline"}):
    """
    Returns a list of tokens delimeted by the strings contained in the
    delimeters set. Punctuation and whitespace characters are treated
    as individual tokens. Delimeters are not counted as tokens.

    Multiple delimeters in a row are counted as a single delimeter.

    string: str
        Some string to be tokenized
    delimeters: set of str
        the delimeters to be used for tokenization
    special_tokens: set of str
        strings that should be treated as individual tokens. Cannot
        contain delimeting characters.
    """
    tokens = []
    s = ""
    for i,char in enumerate(main_string):
        if s in special_tokens:
            tokens.append(s)
            s = ""
        
        if char in delimeters:
            if len(s) > 0:
                tokens.append(s)
            s = ""

        elif char.isalnum():
            s += char
        elif char in string.whitespace:
            if len(s) > 0:
                tokens.append(s)
            tokens.append(char)
            s = ""
        else:
            if len(s) > 0 and s[-1] != "\\":
                tokens.append(s)
                s = ""
            if char == "\\":
                s += char
            else:
                tokens.append(char)
    if len(s) > 0:
        tokens.append(s)
    return tokens

def group_sentences(document, delimeters={".","!","?"},
                  titles={"dr","mr","mrs","ms","prof"}):
    """
    Groups a document string into a list of sentence strings.
    Sentences are defined by a period followed by a whitespace
    character. Handles abbreviations by assuming all abbreviations
    are single, capital characters.

    Input:
        document: str
        delimeters: set of str
            end of sentence characters.
        titles: set of str
            enumerated titles that should be considered.
    Returns:
        sentences: list of str
            a list of all of the sentences in the document.
    """
    sentences = []
    running_s = document[0]
    document = document[1:]
    for i,c in enumerate(document[:-2]):
        running_s += c
        if c in delimeters:
            prob_dec = not document[i+1].isspace()
            other = (not document[i+2].isupper())
            other = other and not document[i-1].isalnum()
            prob_abbrev = document[i-1].isupper()
            prob_title =check_if_title(running_s[:-1],titles)
            if not(prob_dec or prob_abbrev or prob_title or other):
                running_s = running_s.strip()
                if len(running_s) > 0:
                    sentences.append(running_s)
                running_s = ""
    running_s = (running_s+document[-2:]).strip()
    if len(running_s) > 0:
        sentences.append(running_s)
    return sentences

def check_if_title(s, titles):
    """
    A helper function to check if the last word in the string is
    contained within a set of strings.

    s: str
        the string to determine if the last sequence of characters,
        delimeted by whitespace, is in the set `titles`
    titles: set of str
        the titles that need to be compared against
    """
    prob_title = False
    for ws_char in string.whitespace:
        splt = s.strip().split(ws_char)
        idx = -1 if len(splt) > 1 else 0
        title_str = splt[idx].strip().lower()
        prob_title = prob_title or title_str in titles
    return prob_title

