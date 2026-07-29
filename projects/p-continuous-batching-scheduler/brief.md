# Планувальник, який тримає обіцянку

Наш інференс-сервіс обслуговує запити по одному. Черга росте, картка простоює між
запитами, а на піку користувачі бачать хвилинні затримки. Продукт обіцяв p95 нижче
двох секунд і 40 запитів на секунду; зараз ми не тримаємо ні того, ні іншого.

Рішення відоме: обслуговувати запити разом, доклеюючи нові в уже запущений батч, а KV
тримати блоками, а не суцільним куском на найдовший можливий вихід. Треба це написати.

Готового рушія брати не можна — ми маємо зрозуміти механіку, бо далі під неї
налаштовувати прод.

## Що ти пишеш

Три файли в `sched/`. Решта скелета — стенд і тести, їх міняти можна.

### `sched/allocator.py`

```python
class Allocator:
    def __init__(self, num_blocks: int, block_size: int): ...
    def allocate(self) -> int          # найменший вільний номер блока, ref=1
    def share(self, block: int) -> int  # ref += 1, повертає той самий номер
    def release(self, block: int) -> None   # ref -= 1; при нулі блок вільний
    def free_count(self) -> int
    def register(self, block: int, key: str) -> None
    def lookup(self, key: str) -> int | None
```

Блок вважається вільним лише коли лічильник посилань дійшов нуля. Повторний
`release` після нуля не повинен звільняти блок удруге.

### `sched/policy.py`

```python
def victim(running: list) -> object      # кого витісняти
def should_admit(state: dict) -> bool    # чи впускати наступний запит
```

Конвенція, за якою звіряється грейдер: витісняється послідовність із найбільшою
кількістю токенів (`prompt_len + decoded`), при рівності — з більшим `rid`.
`should_admit` отримує `{"running", "max_seqs", "free_blocks", "blocks_needed"}` і
є **єдиним** місцем, де вирішується ліміт одночасних послідовностей. Планувальник
не має дублювати цю перевірку в себе — інакше політику неможливо ні замінити, ні
перевірити.

Витіснена послідовність при `recompute` втрачає весь прогрес і повертається в
початок черги. Після `max_preemptions` витіснень вона вважається нездійсненною.

### `sched/scheduler.py`

```python
class Scheduler:
    def __init__(self, config: dict): ...
    def add(self, requests: list[dict]) -> None
    def step(self) -> dict
    def run(self, max_steps: int = 100000) -> dict
```

Запит — це `{"rid": str, "arrival": int, "prompt": list[int], "output_len": int}`.

`config` містить: `block_size`, `num_blocks`, `max_batch_tokens`, `max_seqs`,
`chunked_prefill`, `prefix_cache`, `preemption` (`"recompute"` або `"swap"`),
`swap_blocks`, `max_preemptions`, `prefill_cost`, `decode_cost`, `step_overhead`.

`step()` повертає `{"t", "prefill_tokens", "decode_tokens", "running", "blocks_used", "ids"}`,
де `ids` — кортеж ідентифікаторів запитів, які отримали роботу на цьому кроці, у
порядку виконання: спершу ті, кому робили prefill, потім ті, кому робили decode.

`run()` повертає метрики: `finished`, `rejected`, `preemptions`, `steps`,
`prefill_tokens`, `decode_tokens`, `ttft_p50`, `latency_p95`, `throughput`,
`cache_hit_rate`.

## Правила часу

Крок коштує `step_overhead + prefill_tokens * prefill_cost + decode_tokens * decode_cost`.
Годинник цілочисельний. Ніякого реального часу — інакше результат не відтворюється.

## Порядок роботи кроку

1. Впустити тих, кому вистачає блоків і хто не перевищує `max_seqs`.
2. Prefill: кожній ще не префіленій послідовності дати шматок у межах спільного
   токенного бюджету кроку. Без `chunked_prefill` — або весь prompt, або нічого.
3. Decode: кожній префіленій дати один токен, якщо вистачає блоків і бюджету.
4. Якщо блоків не вистачає — витіснити когось. Якщо в роботі лишилась одна
   послідовність і їй усе одно не вистачає, вона нездійсненна: познач `rejected`.

Прогрес обов'язковий. Конфігурація, у якій ніхто ніколи не завершується, — це
дефект планувальника, а не властивість навантаження.

## Як це перевіряється

Еталон рахує грейдер сам, на тих самих трасах. Порівнюються метрики і послідовність
кроків. Останній майлстоун інший: ти пишеш тест, який ловить зламану політику, а
ми навмисно ламаємо політику й дивимось, чи твій тест це побачив.

```
mlsys project start p-continuous-batching-scheduler
mlsys project grade p-continuous-batching-scheduler --milestone 1
```
