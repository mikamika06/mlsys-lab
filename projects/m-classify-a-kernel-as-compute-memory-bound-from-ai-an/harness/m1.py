import ref

def check(workdir):
from roofline.classify import classify_bound

out = {"classifications_matched": 0.0}
correct = 0
total = len(ref.TEST_CASES_M1)
for tc in ref.TEST_CASES_M1:
    got = classify_bound(tc["ai"], tc["ridge"])
    if got == tc["want"]:
        correct += 1
    else:
        out["_note"] = f"ai {tc['ai']}, ridge {tc['ridge']}: got {got}, want {tc['want']}"
if correct == total:
    out["classifications_matched"] = 1.0
return out
