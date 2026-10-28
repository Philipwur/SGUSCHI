"""Shared INCAR edits preserve unrelated settings and remain repeatable."""
from utils.Incar import IncarValues, PatchIncar


def TestPatchHandlesMixedCaseDuplicatesWithoutChangingOtherTags():
    Text = "# NSW=9; SMASS=8\nnsw=1; NSW=2; NWRITE=3 ! keep diagnostic\n"
    Text += "smass=0; ENCUT=400\nNSW_OTHER=7\nPREC=Normal"
    Result = PatchIncar(Text, {"nsw":"200"}, Remove=("sMaSs",))
    assert IncarValues(Result) == {"NWRITE":"3", "ENCUT":"400", "NSW_OTHER":"7", "PREC":"Normal", "NSW":"200"}
    assert "# NSW=9; SMASS=8\n" in Result and "! keep diagnostic" in Result
    assert "NSW_OTHER=7\nPREC=Normal\n" in Result
    assert PatchIncar(Result, {"NSW":"200"}, Remove=("SMASS",)) == Result


def TestNoEditsPreserveOriginalTextExactly():
    Text = "prec = Normal\r\n# no trailing newline"
    assert PatchIncar(Text, {}) == Text


def TestRemovalOnlyPreservesCommentsAndSimilarNames():
    Text = "TEBEG=873; TEBEG_OTHER=1 ! temperature\n! TEBEG=0\n"
    Result = PatchIncar(Text, {}, Remove=("TEBEG",))
    assert IncarValues(Result) == {"TEBEG_OTHER":"1"}
    assert "! temperature" in Result and "! TEBEG=0" in Result
    assert PatchIncar(Result, {}, Remove=("TEBEG",)) == Result


def TestValuesReadCommentsSemicolonsAndLastAssignment():
    assert IncarValues("sigma=.1; EDIFF=1e-4 ! SIGMA=3\nSIGMA=.2 # EDIFF=4\n") == {
        "SIGMA":".2", "EDIFF":"1e-4"}
