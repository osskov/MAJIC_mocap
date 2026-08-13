"""
Everything that turns files on disk into PlateTrials. Nothing outside the build path
imports this package.

That is the whole point of it existing: `src/toolchest/{IMUTrace,WorldTrace,PlateTrial}.py`
now hold data and operations on that data, and construction lives here, so the separation is
checkable by grep rather than by convention.

    xsens.py           Xsens .txt export      -> IMUTrace
    reconstruction.py  marker plates          -> poses, with fault isolation and repair
    assembly.py        traces                 -> synchronized PlateTrials
    alborno.py         a dataset's layout     -> Dict[str, PlateTrial]

`experiments/build_trials.py` is the single entry point that drives them and writes the
per-trial parquets under results/trials/<dataset>/.
"""
