# План KV-кешу для гібридної моделі

Нова модель у нас гібридна: частина шарів дивиться на весь контекст, частина —
лише на вікно останніх токенів. Сервер про це не знає й виділяє кожному шару кеш
на повний контекст. У результаті машина тримає вдвічі менше сесій, ніж мала б, а
на 8k контексту падає по пам'яті там, де на 4k працювала.

Треба порахувати, скільки насправді потрібно, і показати це числом.

## Що ти пишеш

`kvplan/groups.py` — `build_groups(config) -> list[group]`. Конфігурація — це
`{"layers": [{"index", "kind", "window", "kv_heads", "head_dim"}, ...]}`, де
`kind` це `"full"` або `"sliding"`. Шари, яким потрібен однаковий кеш, ідуть в
одну групу. Група — це `{"kind", "window", "kv_heads", "head_dim", "layers"}`,
де `layers` — відсортовані індекси, а `window` для повної уваги дорівнює 0.
Групи впорядковані за своїм ключем.

`kvplan/memory.py`:

```python
group_bytes(group, max_context, block_size, bytes_per_element)
plan_bytes(config, max_context, block_size, bytes_per_element)
uniform_bytes(config, max_context, block_size, bytes_per_element)
```

На токен шар тримає ключі **і** значення: `2 · kv_heads · head_dim · bytes_per_element`.
Пам'ять береться блоками, тобто округлюється вгору до `block_size`. Шар з вікном
не може потребувати більше, ніж повний контекст.

`kvplan/schedule.py` — `free_schedule(window, block_size, steps)`. На кожному
кроці `t` від 1 до `steps` поверни, **скільки блоків уже можна звільнити**: усе,
що вийшло за вікно. Число не може зменшуватись із кроками.

## Як перевіряється

Грейдер рахує еталон сам, із тієї ж конфігурації, на трьох різних моделях і двох
розмірах контексту. Третій майлстоун — твій: пишеш тест, а ми підміняємо
групування на таке, що зливає шари з різними вікнами в одну групу. Твій тест має
це побачити.

```
mlsys project start m-build-kv-cache-groups-from-a-hybrid-model-config
mlsys project grade m-build-kv-cache-groups-from-a-hybrid-model-config --milestone 1
```
