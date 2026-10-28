"""Small text helpers shared by MD and bulk-relaxation input preparation."""
import re


def IncarValues(Text):
    """Read active tags, case-insensitively; the last assignment wins."""
    Result = {}
    for Line in Text.splitlines():
        for Part in re.split(r"[#!]", Line, maxsplit=1)[0].split(";"):
            if "=" in Part:
                Key, Value = Part.split("=", 1)
                Result[Key.strip().upper()] = Value.strip()
    return Result


def PatchIncar(Text, Settings, Remove=()):
    """Replace/remove only named tags, retaining other assignments and comments.

    Remove all active occurrences of affected tags, including mixed-case and
    semicolon-separated duplicates, then append one assignment per replacement.
    File writing and guards against modifying started jobs belong to callers.
    """
    Settings = {Tag.upper(): Value for Tag, Value in Settings.items()}
    Tags = set(Settings) | {Tag.upper() for Tag in Remove}
    if not Tags:
        return Text
    ReplaceTag = re.compile(r"\s*(?:" + "|".join(re.escape(Tag) for Tag in sorted(Tags)) + r")\s*=", re.I)
    Lines = []
    for Line in Text.splitlines(keepends=True):
        Parts = re.split(r"([#!].*)", Line.rstrip("\r\n"), maxsplit=1)
        Assignments = Parts[0].split(";")
        Kept = [Part for Part in Assignments if not ReplaceTag.match(Part)]
        if Kept == Assignments:
            Lines.append(Line)
        else:
            Code = ";".join(Kept).rstrip()
            Comment = Parts[1] if len(Parts) > 1 else ""
            if Code.strip() or Comment:
                Lines.append(Code + (" " if Code and Comment else "") + Comment + "\n")
    Result = "".join(Lines)
    if Result and not Result.endswith("\n"):
        Result += "\n"
    return Result + "".join("{} = {}\n".format(Tag, Value) for Tag, Value in Settings.items())
