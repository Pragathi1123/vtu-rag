import csv
import time
from rag import answer, retrieve

BUSY = "The AI service is busy"

with open("questions.txt", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

with open("results.csv", "w", newline="", encoding="utf-8") as out:
    w = csv.writer(out)
    w.writerow(["No", "Question", "Best distance", "Answer", "Sources",
                "Answer correct (Y/N)", "Page correct (Y/N)"])
    for i, q in enumerate(questions, 1):
        best = min(h[2] for h in retrieve(q))
        r = answer(q)
        for _ in range(2):                        # extra retries if still busy
            if not r["answer"].startswith(BUSY):
                break
            print("  still busy, waiting 30s...", flush=True)
            time.sleep(30)
            r = answer(q)
        srcs = ", ".join(f"p.{s['page']}" for s in r["sources"])
        w.writerow([i, q, round(best, 3), r["answer"].replace("\n", " "), srcs, "", ""])
        out.flush()
        print(i, q, "->", round(best, 3), "|", r["answer"][:60].replace("\n", " "), flush=True)
        time.sleep(3)

print("Saved results.csv")
