"""Poll :37565 until all heavy 100%-CPU workers (rawmse / lossH / sim_fake_worm)
exit, then launch _baai_ablation.py. Adaptive 30-min sleep."""
import sys, time, json
sys.path.insert(0, r"D:\Warm")
from _ssh_helper import run

POLL_SEC = 30 * 60
KEYWORDS = ["rawmse_worker", "lossH_baai_worker", "sim_fake_worm_gen"]


def busy_count():
    r = run("37565",
            "ps -eo pid,etime,pcpu,cmd --sort=-pcpu | grep -E 'python' "
            "| grep -v grep")
    out = r.get("stdout", "")
    n = 0
    for line in out.splitlines():
        if any(k in line for k in KEYWORDS):
            # only count lines that show high cpu (>=50)
            parts = line.split()
            try:
                pcpu = float(parts[2])
            except Exception:
                continue
            if pcpu >= 50:
                n += 1
    return n, out


def launch_ablation():
    cmd = ("cd /root && nohup /root/miniconda3/envs/py38/bin/python -u "
           "/root/_baai_ablation.py --levels 0,1,2,3,4,5 --seeds 0,1,2 "
           "--out /root/baaiworm_ablation_claim4.json "
           "> /root/baai_ablation.log 2>&1 & echo PID=$!")
    r = run("37565", cmd, timeout=30)
    return r


def main():
    started = False
    while True:
        n, snap = busy_count()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] busy={n}", flush=True)
        if n == 0:
            print("[LAUNCH] firing BAAIWorm ablation", flush=True)
            r = launch_ablation()
            print("LAUNCH RESULT:", r.get("stdout"), r.get("stderr"))
            started = True
            break
        time.sleep(POLL_SEC)
    # follow-up: poll output file growth
    if started:
        for _ in range(80):  # up to ~6h after launch
            time.sleep(POLL_SEC)
            r = run("37565",
                    "tail -5 /root/baai_ablation.log 2>&1; "
                    "echo ---; ls -la /root/baaiworm_ablation_claim4.json 2>&1")
            print(time.strftime("%H:%M:%S"), r.get("stdout"), flush=True)
            # check if ALL DONE printed
            t = run("37565", "grep -c 'ALL DONE' /root/baai_ablation.log 2>/dev/null")
            if "1" in (t.get("stdout") or "").strip():
                print("[FINISHED]", flush=True)
                break


if __name__ == "__main__":
    main()
