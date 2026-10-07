"""pandas <-> PySpark conversion with Apache Arrow OFF vs ON, visualized with rich.

pip install pyspark pandas pyarrow rich psutil numpy
python arrow_demo.py --rows 200000
"""
import argparse, threading, time, warnings

import numpy as np
import pandas as pd
import psutil
from pyspark.sql import SparkSession

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()
ARROW = "spark.sql.execution.arrow.pyspark.enabled"
BARS = "▁▂▃▄▅▆▇█"


# ---------- resource monitor (Python process + JVM child, sampled in a thread) ----------
class Monitor:
    def __init__(self, interval=0.05):
        self.interval, self.cpu, self.mem = interval, [], []
        self._stop, self._procs = threading.Event(), {}
        self.base = self._rss()

    def _tree(self):
        me = psutil.Process()
        return [me] + me.children(recursive=True)  # JVM runs as a child process

    def _rss(self):
        return sum(p.memory_info().rss for p in self._tree() if p.is_running())

    def _loop(self):
        while not self._stop.is_set():
            cpu = mem = 0
            for p in self._tree():
                try:
                    first = p.pid not in self._procs
                    proc = self._procs.setdefault(p.pid, p)
                    c = proc.cpu_percent(None)  # first call always returns 0.0
                    cpu += 0 if first else c
                    mem += p.memory_info().rss
                except psutil.Error:
                    pass
            self.cpu.append(cpu)
            self.mem.append(mem / 2**20)
            time.sleep(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()


def sparkline(vals, width=60):
    vals = vals[-width:]
    hi = max(vals, default=1) or 1
    return "".join(BARS[min(7, int(v / hi * 7))] for v in vals)


def run_case(title, fn):
    """Run fn in a worker thread while showing live CPU/memory sparklines."""
    out = {}
    worker = threading.Thread(target=lambda: out.update(result=fn()))
    with Monitor() as mon, Live(console=console, refresh_per_second=10, transient=True) as live:
        t0 = time.perf_counter()
        worker.start()
        while worker.is_alive():
            if mon.cpu:
                live.update(Panel(
                    f"[cyan]CPU {mon.cpu[-1]:6.0f}%[/] {sparkline(mon.cpu)}\n"
                    f"[magenta]MEM {mon.mem[-1]:6.0f}MB[/] {sparkline(mon.mem)}\n"
                    f"[green]TIME {time.perf_counter() - t0:5.1f}s[/]",
                    title=f"▶ {title}", border_style="yellow"))
            time.sleep(0.1)
        elapsed = time.perf_counter() - t0
    mb = [m - mon.base / 2**20 for m in mon.mem] or [0]
    return dict(time=elapsed, peak_mem=max(mb), avg_cpu=float(np.mean(mon.cpu or [0])), peak_cpu=max(mon.cpu or [0]), result=out.get("result"))

# ---------- data + helpers ----------
def make_pandas(n):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id": np.arange(n, dtype="int64"),
        "value": rng.normal(size=n),
        "category": rng.choice(["alpha", "beta", "gamma", "delta"], n),
        "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 10**6, n), unit="s"),
        "flag": rng.random(n) > 0.5,
    })


def show_data(pdf, sdf):
    tbl = Table(title="Source pandas sample", header_style="bold cyan")
    for c in pdf.columns:
        tbl.add_column(f"{c}\n[dim]{pdf[c].dtype}[/]")
    for row in pdf.head(3).itertuples(index=False):
        tbl.add_row(*[str(v) for v in row])
    console.print(tbl)

    # round-trip dtype mapping, measured for real in both modes
    spark = sdf.sparkSession
    back = {}
    for flag in ("false", "true"):
        spark.conf.set(ARROW, flag)
        back[flag] = sdf.limit(100).toPandas().dtypes.astype(str)
    m = Table(title="Type mapping: pandas → Spark → pandas", header_style="bold magenta")
    for col in ("column", "pandas", "Spark type", "back (Arrow OFF)", "back (Arrow ON)"):
        m.add_column(col)
    spark_types = {f.name: f.dataType.simpleString() for f in sdf.schema.fields}
    for c in pdf.columns:
        diff = back["false"][c] != back["true"][c]
        m.add_row(c, str(pdf[c].dtype), spark_types[c], back["false"][c], f"[bold yellow]{back['true'][c]}[/]" if diff else back["true"][c])
    console.print(m)


def show_results(rows):
    t = Table(title="Results", header_style="bold green")
    for col in ("Direction", "Arrow", "Time (s)", "Speedup", "ΔMem (MB)", "Avg CPU%", "Peak CPU%"):
        t.add_column(col, no_wrap=True, justify="right" if col not in ("Direction", "Arrow") else "left")
    slowest = {}
    for r in rows:
        slowest[r["dir"]] = max(slowest.get(r["dir"], 0), r["time"])
    for r in rows:
        on = r["arrow"] == "ON"
        t.add_row(r["dir"], "[green]ON[/]" if on else "[red]OFF[/]", f"{r['time']:.2f}",
                  f"[bold green]{slowest[r['dir']] / r['time']:.1f}x[/]" if on else "1.0x",
                  f"{r['peak_mem']:.0f}", f"{r['avg_cpu']:.0f}", f"{r['peak_cpu']:.0f}")
    console.print(t)
    hi = max(r["time"] for r in rows)
    for r in rows:  # tiny bar chart of wall time
        n = max(1, int(r["time"] / hi * 50))
        colour = "green" if r["arrow"] == "ON" else "red"
        console.print(f"{r['dir']:<16} {r['arrow']:<3} [{colour}]{'█' * n}[/] {r['time']:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    n = ap.parse_args().rows

    spark = (SparkSession.builder.master("local[*]").appName("arrow-demo")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    with console.status("Generating pandas data + warming up the JVM..."):
        pdf = make_pandas(n)
        spark.conf.set(ARROW, "true")
        sdf = spark.createDataFrame(pdf).cache()
        sdf.count()  # materialize so toPandas() measures transfer only
    console.print(f"[bold]{n:,} rows × {pdf.shape[1]} cols, {pdf.memory_usage(deep=True).sum() / 2**20:.0f} MB in pandas[/]\n")
    show_data(pdf, sdf)

    results = []
    for direction, fn in (("pandas → Spark", lambda: spark.createDataFrame(pdf).count()), ("Spark → pandas", lambda: len(sdf.toPandas()))):
        for flag in ("false", "true"):
            spark.conf.set(ARROW, flag)
            label = "ON" if flag == "true" else "OFF"
            r = run_case(f"{direction}  |  Arrow {label}", fn)
            results.append({**r, "dir": direction, "arrow": label})
    show_results(results)
    spark.stop()


if __name__ == "__main__":
    main()