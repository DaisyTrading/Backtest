export QUANDL_API_KEY="f1Zt4NxabDFaUpzXu4Wa"
zipline ingest -b quandl
cd /Users/yuweiyan/IdeaProjects/DaisyTrading/Backtest/Imp
zipline run -f dual_moving_average.py --start 2014-1-1 --end 2020-1-1 -o dma.pickle --no-benchmark
python show_performance.py