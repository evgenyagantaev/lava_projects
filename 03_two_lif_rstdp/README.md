# Two LIF neurons with R-STDP

Минимальный пример на lava-nc: два LIF нейрона, между ними обучаемый `LearningDense` с правилом `RewardModulatedSTDP`. На пресинаптический нейрон подаются случайные спайки, вознаграждение приходит в случайные моменты времени. Все элементы сети взяты из классов lava-nc (LIF, LearningDense, RewardModulatedSTDP, вспомогательный RSTDPLIF из туториала lava-nc).

Запуск:
```
python projects/03_two_lif_rstdp/main.py
```
