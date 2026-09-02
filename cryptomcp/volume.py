"""Объёмные метрики с поправкой на суточную сезонность (PLAN §4.10).

У крипты сильная суточная сезонность: свеча в 04:00 UTC систематически тише,
чем в 14:00. Сравнение с плоской MA20 на внутридневных таймфреймах показывало бы
«затухание объёма 0.6x» там, где на самом деле просто ночь — то есть врало бы
системно, а не изредка.

Это касается и 4h: в сутках шесть четырёхчасовых свечей, и ночная из них тише
дневных. Ограничиться «только 4h и 1d» проблему не решает.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .series import Series, interval_ms

DAY_MS = 86_400_000

#: Таймфреймы, на которых суточной сезонности нет по построению.
_NO_SEASONALITY = {"1d", "3d", "1w"}

#: Минимум наблюдений на слот, ниже которого baseline объявляется слабым.
MIN_SAMPLES_PER_SLOT = 10

#: Минимум, ниже которого объёмные метрики не участвуют в скоринге вовсе.
MIN_SAMPLES_TO_SCORE = 5


@dataclass(frozen=True)
class VolumeContext:
    """Текущий объём относительно сопоставимой базы."""

    #: Отношение объёма последней свечи к базе.
    ratio: float
    #: Как считалась база: "сезонный слот" или "скользящая средняя".
    basis: str
    #: Наблюдений в базе.
    samples: int
    #: Отношение MA20 к MA100 — затухание объёма из ТЗ §4.2, группа 2.
    ma_ratio: float
    #: Число аномальных баров за последние 30 свечей.
    anomalous_bars: int
    #: Средняя доля тейкер-покупок за последние 30 свечей.
    taker_buy_mean: float

    @property
    def weak_basis(self) -> bool:
        return self.samples < MIN_SAMPLES_PER_SLOT

    @property
    def scorable(self) -> bool:
        """Участвует ли объёмная группа в squeeze_index.

        На 5m и 1m наблюдений на слот суток набирается единицы, и база
        перестаёт что-либо значить. Лучше исключить группу из скоринга, чем
        подмешать в индекс шум (PLAN §4.10).
        """
        return self.samples >= MIN_SAMPLES_TO_SCORE

    def to_dict(self) -> dict[str, object]:
        return {
            "ratio": round(self.ratio, 3),
            "basis": self.basis,
            "samples": self.samples,
            "weak_basis": self.weak_basis,
            "ma20_over_ma100": round(self.ma_ratio, 3),
            "anomalous_bars_30": self.anomalous_bars,
            "taker_buy_mean_30": round(self.taker_buy_mean, 3),
        }


def slot_of_day(open_time_ms: np.ndarray, interval: str) -> np.ndarray:
    """Номер слота внутри суток для каждой свечи."""
    step = interval_ms(interval)
    return ((open_time_ms % DAY_MS) // step).astype("int64")


#: Окно для оценки текущего уровня объёма, свечей.
LEVEL_WINDOW = 20


@dataclass(frozen=True)
class Baseline:
    """База сравнения объёма, посчитанная по свечам ряда.

    Одна конструкция обслуживает и снапшот, и сырые свечи. Раньше их было две:
    снапшот сравнивал свечу с сезонной базой, а get_klines — со средним по
    показанному окну, из-за чего одна и та же свеча получала разные числа в
    похоже названных колонках, а число в get_klines вдобавок зависело от
    параметра limit (30.08 CAKE: 4.10x при limit=50 и 2.25x при limit=14).
    """

    #: База на каждую свечу ряда; nan там, где она не считалась.
    values: np.ndarray
    #: Наблюдений в базе на каждую свечу.
    samples: np.ndarray
    #: Как считалась база последней свечи ряда.
    basis: str

    def ratio(self, volumes: np.ndarray) -> np.ndarray:
        """Отношение объёма к базе для каждой свечи."""
        return np.divide(
            volumes, self.values,
            out=np.full(len(volumes), np.nan), where=self.values > 0,
        )


def baseline_series(series: Series, count: int = 1) -> Baseline:
    """База для сравнения объёма: свежий уровень, поправленный на форму суток.

    Считается для последних ``count`` свечей ряда; для остальных в values стоит
    nan. Ограничение по count здесь ради стоимости: медиана истории на каждую
    свечу по ряду в несколько тысяч свечей ощутима, а нужны всегда либо одна
    последняя свеча (снапшот), либо показываемый хвост (get_klines).

    Наивная медиана того же слота за всё окно смешивает два разных эффекта:
    форму суток и общий дрейф активности. Измерено на живых данных: у SUIUSDT
    медиана слота в первой половине шестидесятидневного окна 1.97M, во второй
    3.16M. При такой базе обычная свеча показывала бы 4x, и метрика читалась бы
    как всплеск, хотя измеряла бы всего лишь «сегодня оживлённее, чем два
    месяца назад».

    Поэтому эффекты разделены. Форма суток — устойчивое отношение медианы слота
    к общей медиане, её имеет смысл брать с длинного окна. Уровень — медиана
    последних свечей. База есть произведение одного на другое.

    Дрейф уровня при этом не теряется: он измеряется отдельной метрикой
    MA20/MA100, где он и является предметом (ТЗ §4.2, группа 2).
    """
    volumes = series.quote_volume
    n = len(volumes)
    values = np.full(n, np.nan)
    samples = np.zeros(n, dtype="int64")
    if n < 2:
        return Baseline(values, samples, "недостаточно данных")

    seasonal = series.interval not in _NO_SEASONALITY
    slots = (
        slot_of_day(series.df["open_time"].to_numpy(dtype="int64"), series.interval)
        if seasonal else np.zeros(n, dtype="int64")
    )

    basis = "недостаточно данных"
    for i in range(max(1, n - count), n):
        # Свеча в свою базу не входит: сравнивать её с самой собой нельзя.
        recent = volumes[max(0, i - LEVEL_WINDOW):i]
        level = float(np.median(recent))
        history = volumes[:i]
        same_slot = history[slots[:i] == slots[i]] if seasonal else np.empty(0)
        overall = float(np.median(history)) if len(history) else float("nan")

        if not seasonal:
            values[i], samples[i] = level, len(recent)
            note = f"медиана последних {len(recent)}"
        elif len(same_slot) == 0 or not overall or np.isnan(overall):
            values[i], samples[i] = level, len(recent)
            note = f"медиана последних {len(recent)} (слот не набран)"
        else:
            factor = float(np.median(same_slot)) / overall
            values[i], samples[i] = level * factor, len(same_slot)
            hours = int(slots[i]) * interval_ms(series.interval) / 3_600_000
            note = (
                f"уровень последних {len(recent)} × "
                f"сезонность слота {hours:04.1f} UTC ({factor:.2f})"
            )
        if i == n - 1:
            basis = note

    return Baseline(values, samples, basis)


def seasonal_baseline(series: Series) -> tuple[float, str, int]:
    """База сравнения для последней свечи ряда: (база, описание, наблюдений)."""
    baseline = baseline_series(series)
    if len(baseline.values) == 0:
        return float("nan"), baseline.basis, 0
    return float(baseline.values[-1]), baseline.basis, int(baseline.samples[-1])


def anomalous_bars(series: Series, atr_values: np.ndarray, *, window: int = 30,
                   volume_multiple: float = 3.0, move_atr_fraction: float = 0.5) -> int:
    """Бары с большим объёмом и малым движением цены (ТЗ §4.2, группа 2).

    Интерпретация из ТЗ: набор позиции без движения цены. Объём сравнивается
    со скользящей средней, движение — с ATR, чтобы порог был сопоставим с
    волатильностью инструмента, а не задан в процентах на все случаи.
    """
    volumes = series.quote_volume
    if len(volumes) < window + 20:
        return 0

    rolling_mean = pd.Series(volumes).rolling(20).mean().to_numpy()
    opens = series.col("open")
    closes = series.close

    count = 0
    for i in range(len(volumes) - window, len(volumes)):
        mean = rolling_mean[i]
        atr_value = atr_values[i] if i < len(atr_values) else np.nan
        if np.isnan(mean) or np.isnan(atr_value) or mean <= 0 or atr_value <= 0:
            continue
        big_volume = volumes[i] >= volume_multiple * mean
        small_move = abs(closes[i] - opens[i]) < move_atr_fraction * atr_value
        if big_volume and small_move:
            count += 1
    return count


def volume_context(series: Series, atr_values: np.ndarray) -> VolumeContext:
    """Полная объёмная картина по ряду."""
    volumes = series.quote_volume
    baseline, basis, samples = seasonal_baseline(series)

    ratio = float(volumes[-1] / baseline) if baseline and not np.isnan(baseline) else float("nan")

    ma20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float("nan")
    ma100 = float(np.mean(volumes[-100:])) if len(volumes) >= 100 else float("nan")
    ma_ratio = ma20 / ma100 if ma100 else float("nan")

    taker = series.taker_buy_ratio[-30:]
    taker_mean = float(np.nanmean(taker)) if len(taker) else float("nan")

    return VolumeContext(
        ratio=ratio,
        basis=basis,
        samples=samples,
        ma_ratio=ma_ratio,
        anomalous_bars=anomalous_bars(series, atr_values),
        taker_buy_mean=taker_mean,
    )


#: Нейтраль доли тейкер-покупок: половина объёма прошла по рынку в покупку.
TAKER_NEUTRAL = 0.50

#: Выше этого доля считается перевесом покупателя, а не шумом вокруг нейтрали.
TAKER_PRESSURE = 0.55

#: Окно, на котором меряется поглощение. То же, что у аномальных баров.
ABSORPTION_WINDOW = 30


#: Во сколько раз объём должен превысить базу, чтобы считаться всплеском.
LEAD_VOLUME_MULTIPLE = 3.0

#: Насколько тело свечи должно превысить ATR, чтобы считаться движением цены.
LEAD_MOVE_ATR = 1.5

#: Всплеск объёма засчитывается за набор, только если сама свеча тихая — тот же
#: порог, что у баров набора. Замерено: без этого условия признак не различает
#: два кейса, ради которых заведён (ASTER +8 свечей против UAI +2, знак один).
QUIET_BAR_ATR = 0.5


@dataclass(frozen=True)
class Absorption:
    """Признаки поглощения на МЛАДШЕМ таймфрейме.

    Отдельная от `VolumeContext` величина, потому что считается по другому ряду.
    Сжатие измеряется на своём таймфрейме, накопление — всегда на часовом или
    мельче, и вот почему: у ASTER перед пробоем 19.08 дневная свеча показывала
    ноль аномальных баров и долю тейкер-покупок 0.48, то есть «покупателя нет».
    Всё поглощение произошло ВНУТРИ этой свечи — 7.15x объёма при теле +0.03%
    в 06:00 и семь часов повышенного объёма при стоящей цене, с долей
    тейкер-покупок до 0.72. Определение аномального бара при этом одно и то же;
    разное только разрешение, и дневное стирает признак полностью.

    Средней по окну для той же причины мало: 0.48 на дневке прячет часы по
    0.57, 0.60 и 0.72. Серия важнее разового выброса — она означает устойчивый
    перевес покупателя, а не одну заявку.
    """

    interval: str
    bars: int
    window: int
    taker_mean: float
    taker_max: float
    taker_above: int
    taker_streak: int
    volume_ratio: float
    weak_basis: bool
    #: На сколько свечей всплеск объёма опередил движение цены; None — когда
    #: одного из двух событий в окне не было. Словесное состояние обязательно:
    #: голое число не отличает «ещё не разрешилось» от «объём пришёл позже».
    lead_bars: int | None = None
    lead_state: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "interval": self.interval,
            "absorption_bars": self.bars,
            "window": self.window,
            "taker_mean": round(self.taker_mean, 3),
            "taker_max": round(self.taker_max, 3),
            "taker_above_pressure": self.taker_above,
            "taker_longest_streak": self.taker_streak,
            "volume_ratio": round(self.volume_ratio, 3),
            "volume_lead_bars": self.lead_bars,
            "volume_lead_state": self.lead_state,
        }


def longest_streak(values: np.ndarray, threshold: float) -> int:
    """Самая длинная серия подряд выше порога."""
    best = current = 0
    for value in values:
        if not np.isnan(value) and value > threshold:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def absorption(
    series: Series, atr_values: np.ndarray, *, window: int = ABSORPTION_WINDOW
) -> Absorption:
    """Поглощение по младшему ряду: бары набора и разрешение по тейкерам."""
    taker = series.taker_buy_ratio[-window:]
    lead_bars, lead_state = volume_leads_price(series, atr_values, window=window)
    volumes = series.quote_volume
    baseline, _, samples = seasonal_baseline(series)
    ratio = (
        float(volumes[-1] / baseline)
        if baseline and not np.isnan(baseline) else float("nan")
    )
    return Absorption(
        interval=series.interval,
        bars=anomalous_bars(series, atr_values, window=window),
        window=min(window, len(taker)),
        taker_mean=float(np.nanmean(taker)) if len(taker) else float("nan"),
        taker_max=float(np.nanmax(taker)) if len(taker) else float("nan"),
        taker_above=int(np.sum(taker > TAKER_PRESSURE)) if len(taker) else 0,
        taker_streak=longest_streak(taker, TAKER_NEUTRAL),
        volume_ratio=ratio,
        weak_basis=samples < MIN_SAMPLES_PER_SLOT,
        lead_bars=lead_bars,
        lead_state=lead_state,
    )


def _candles(count: int) -> str:
    """«1 свечу», «2 свечи», «5 свечей» — иначе выдача читается как машинная."""
    tail = abs(count) % 10
    hundred = abs(count) % 100
    if tail == 1 and hundred != 11:
        return f"{count} свечу"
    if tail in (2, 3, 4) and hundred not in (12, 13, 14):
        return f"{count} свечи"
    return f"{count} свечей"


def volume_leads_price(
    series: Series,
    atr_values: np.ndarray,
    *,
    window: int = ABSORPTION_WINDOW,
    volume_multiple: float = LEAD_VOLUME_MULTIPLE,
    move_atr: float = LEAD_MOVE_ATR,
) -> tuple[int | None, str]:
    """На сколько свечей всплеск объёма опередил движение цены.

    Признак, отличающий набор позиции от реакции на событие. Замерено на двух
    случаях с противоположным исходом:

    - ASTER 19.08: пик объёма 7.15x в 06:00 при теле +0.03%, первое движение
      цены только к вечеру — объём пришёл ЗАРАНЕЕ, и это был набор.
    - UAI 29.08: объём 3.44x пришёл той же свечой, что и тело +2.75%, а пик
      10.60x — уже после того, как цена прошла своё. Объём шёл ЗА ценой.

    Оба выглядят как «всплеск объёма», и без этой разницы они неразличимы.

    Возвращается пара «сколько свечей» и словесное состояние. Состояний пять,
    и четыре из них — не число:

    - всплеска объёма в окне не было;
    - всплеск был, движения ещё не было — самое интересное для сканера
      состояние, потому что развязка впереди; число тогда означает, сколько
      свечей прошло с всплеска, и это нижняя граница, а не итог;
    - движение было, всплеска перед ним не было;
    - оба были: знак разницы и есть ответ.

    База объёма — скользящая средняя двадцати свечей, та же, что у баров
    набора. Сезонная база (§4.10) точнее, но тогда два числа в одном блоке
    считались бы от разных величин, и сравнивать их стало бы нельзя.
    """
    volumes = series.quote_volume
    if len(volumes) < window + 20:
        return None, "n/a (истории меньше окна)"

    rolling = pd.Series(volumes).rolling(20).mean().to_numpy()
    opens, closes = series.col("open"), series.close
    start = len(volumes) - window

    first_volume: int | None = None
    first_move: int | None = None
    for i in range(start, len(volumes)):
        mean, atr_value = rolling[i], atr_values[i] if i < len(atr_values) else np.nan
        if np.isnan(mean) or np.isnan(atr_value) or mean <= 0 or atr_value <= 0:
            continue
        quiet = abs(closes[i] - opens[i]) < QUIET_BAR_ATR * atr_value
        if first_volume is None and quiet and volumes[i] >= volume_multiple * mean:
            first_volume = i
        if first_move is None and abs(closes[i] - opens[i]) >= move_atr * atr_value:
            first_move = i

    if first_volume is None and first_move is None:
        return None, "ни всплеска объёма, ни движения цены"
    if first_volume is None:
        return None, "всплеск объёма был движением цены, а не набором"
    if first_move is None:
        waited = len(volumes) - 1 - first_volume
        return waited, f"всплеск объёма {_candles(waited)} назад, движения ещё не было"

    lead = first_move - first_volume
    if lead > 0:
        return lead, f"объём опередил цену на {_candles(lead)}"
    if lead == 0:
        return 0, "объём и движение одной свечой"
    return lead, f"объём пришёл на {_candles(-lead)} ПОЗЖЕ движения"
