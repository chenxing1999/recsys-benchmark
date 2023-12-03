from src.dataset.avazu.utils import run_timestamp_preprocess


def test_timestamp_preprocess():
    line = "1000009418151094273,0,14102100,1005,0,1fbe01fe,f3845767,28905ebd,ecad2386,7801e8d9,07d7df22,a99f214a,ddd2926e,44956a24,1,2,15706,320,50,1722,0,35,-1,79"  # noqa
    values = line.rstrip("\n").split(",")

    hour, weekday, is_weekend = run_timestamp_preprocess(values)
    assert hour == "0"
    assert weekday == "1"
    assert is_weekend == "False"
