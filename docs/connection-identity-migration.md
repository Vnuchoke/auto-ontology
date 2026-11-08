<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Connection identity migration

## Connection API contract

Management payloads now keep `connection_id`, `display_label`, `database_name`
and sanitized `connection` settings separate. The physical name is not unique.
Create accepts `{connection, display_label?}`; an omitted label defaults to the
physical name only when free. Empty labels are invalid, normalized collisions
return 409, and credentials/references never appear in returned settings.

Use `PUT` / `DELETE /api/connections/by-id/{connection_id}` and
`PATCH /api/connections/by-id/{connection_id}/sso-federation` for mutations.
`PUT` accepts `{display_label}` alone without credential reads, connectivity
tests or ingestion. Physical database and connector type cannot change. Blank
secret fields on a settings update retain that ID's values.

`POST /api/connections/by-id/{connection_id}/test` tests proposed `{connection}`
using that source's retained secrets. The existing `/api/connections/test`
also accepts `replacing_connection_id`; its legacy `replacing` selector is
mutually exclusive. Other legacy name routes remain available only for exactly
one matching managed source: ambiguity returns 409 before reads or writes to
credentials. Missing IDs return 404, unavailable selected credentials 503.
Unexpected backend exception text is not exposed.

The generic catalog-node patch cannot alter connection IDs, physical names,
labels, origins, bindings or credentials. Change labels through the lifecycle
endpoint so the same normalization and conflict checks apply. These management
endpoints retain their settings permissions through the frontend proxy; the
chat-authorized discovery endpoint is separate and does not return settings.

This contract is part of a coordinated release, not permission to roll out the
partially implemented API while older workers/frontend are still active.

**Не обновлять рабочую среду частично.** Схема identity сама по себе не делает старые API, ingestion или retrieval совместимыми с одноимёнными БД. Нужен согласованный source-aware релиз всех процессов; старые writers во время миграции должны быть остановлены.

## Предпосылки и границы

- Работать из корня репозитория в согласованной среде. Любые production SQL, остановка работающих сервисов, восстановление backup и Vault cleanup требуют отдельного разрешения оператора.
- Передать `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DATABASE` в окружение безопасным способом. Helper не загружает `.env` и не меняет настройки подключения. Для приведённых Alembic-команд также отключается автоматическая загрузка `.env`.
- Alembic владеет backend-схемой `public`; Prisma — только `frontend`; VDB-код — векторными коллекциями. Не добавлять `public` или vector schema в Prisma и не передавать их Alembic как чужие таблицы для удаления.
- `--check` читает metadata и vector identity без writes, Vault, analyzed databases или model calls. Connection JSON и содержимое vectors не читаются. Отчёт содержит ID, коды проблем и счётчики, но не секреты, SQL-текст или credential references. `--apply` после metadata commit может вызвать настроенный embedding API для обязательного semantic rebuild; это требует разрешения на передачу source-scoped semantic text этому сервису.

### Операторский checklist будущего rollout

Локальные проверки ниже **не разрешают** production rollout. Для каждого шага
сначала получить отдельное разрешение на точную среду и сохранить безопасный receipt.

1. **Backup + stop:** проверить восстановление согласованного catalog/vector/frontend
   backup на отдельной БД; остановить всех старых writers, зафиксировать config/versions.
2. **Migrate:** additive revision → read-only check → явный mapping конфликтов → apply
   и повторный apply → semantic rebuild → финальная revision. Не продолжать с issues,
   `semantic_rebuild_failed` или неизвестным источником исполняемого SQL.
3. **Verify:** сверить ID, descriptions, точные credential refs и NULL legacy chats;
   проверить ready выбранного CID, source list/errors/provenance без secrets,
   A/B control queries и неизменность B после разрешённой адресной операции A.
4. **Activate:** включить согласованный source-aware комплект API/ingestion/frontend/MCP
   только после review evidence; проверить env-first и права обычного пользователя.
5. **Rollback:** снова остановить writers; после появления duplicates/tombstones не
   принуждать downgrade и не удалять данные — восстановить согласованный backup либо
   выполнить отдельно согласованный forward fix. Vault cleanup разрешается отдельно,
   только по подтверждённым неиспользуемым ID paths; legacy refs не переносить массово.

## 1. Backup и остановка писателей

Сохранить согласованный backup всей БД: каталог, векторные схемы и `frontend`, включая сообщения. Проверить список объектов backup и возможность восстановления на отдельной БД; наличие файла без проверки недостаточно. Использовать действующие deployment/backup процедуры, а не переносить команды от другой среды.

Остановить backend API/chat workers, ingestion/semantic schedulers и старый frontend, который может самостоятельно писать сообщения. Не допускать mixed-version workers. Сохранить текущую конфигурацию env-источников и её соответствие metadata; пароль не используется как идентификатор источника.

Legacy Vault paths не переносятся и не удаляются. Содержимое Vault не требуется для определения source ID. Заранее подтвердить точные references известных Vault-managed источников; по NULL `connection` нельзя отличить их от env/catalog-only корней.

## 2. Additive schema

На **разрешённом store** применить только additive revision:

```bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync alembic upgrade c19a83e47b20
```

Revision добавляет nullable label/origin/credential-ref и SQL ownership. Она не меняет существующие root/child IDs и пока сохраняет прежнюю уникальность физических имён. Не запускать `upgrade head` вслепую: следующая revision требует заполненных и проверенных ярлыков.

Изменение `frontend.conversations.connection_id` применять штатным Prisma deployment-путём этого релиза только в схеме `frontend`. Старые записи остаются NULL. Историю нельзя автоматически приписывать сегодняшнему единственному источнику: legacy-диалоги с сообщениями становятся read-only.

## 3. Проверка и явные решения оператора

```bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m dev_tools.migrate_connection_identity --check
```

Код 0 означает отсутствие блокирующих проблем **metadata backfill**, а не готовность запросов. Код 2 означает конфликт/необходимость mapping либо ошибку настройки/схемы; при ошибке соединения подробности с возможными секретами не печатаются. Сопоставлять ID с известными источниками через разрешённые административные средства.

JSON mapping принимает только `display_label`, `configuration_origin`, `credential_ref`, с ключом, в точности равным существующему ID. Пример использует синтетический ID и условный legacy reference:

```json
{
  "11111111-1111-4111-8111-111111111111": {
    "display_label": "sales-local",
    "configuration_origin": "managed",
    "credential_ref": "sales"
  }
}
```

Не помещать пароли, токены или connection strings в mapping. Reference сохраняется точно, без нормализации пути. Отсутствующий ярлык заполняется физическим именем; конфликт `lower(TRIM(label))` требует другого явно выбранного ярлыка, а не случайного суффикса. Managed inline connection определяется без чтения его значений. Неизвестный origin требует явного решения; `configuration_origin: null` обозначает подтверждённый catalog-only корень, который не становится активным подключением. Существующий managed источник не переводится в env и не перезаписывает собственный reference ради совпавшего имени.

```bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m dev_tools.migrate_connection_identity --check --mapping identity-mapping.json
```

`issues` блокируют весь apply; частичного обновления допустимых строк нет. `unresolved_sql` перечисляет определения без доказанного единственного владельца. Они сохраняются с NULL ownership и должны быть исключены source-aware кодом из retrieval, SQL examples и prompt rules. Не назначать владельца только по тексту SQL или имени атрибута.

## 4. Metadata backfill

```bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m dev_tools.migrate_connection_identity --apply --mapping identity-mapping.json
PYTHON_DOTENV_DISABLED=1 uv run --no-sync python -m dev_tools.migrate_connection_identity --check --mapping identity-mapping.json
```

Необязательный `--mapping` можно опустить, если preflight не требует решений. Root/SQL metadata и vector identity backfill выполняются одной транзакцией с advisory lock. Catalog IDs, описания, vector content/embedding и secrets не переписываются; vector primary keys переводятся на детерминированные UUIDv5. Повторный apply идемпотентен. Metadata `pending_updates: 0` сам по себе не означает завершённую vector migration.

SQL ownership выводится из полного набора table/column dependencies; SQL attributes и custom analyses наследуют его лишь при однозначности всех связанных запросов. Существующее явно заданное ownership не перезаписывается при конфликте зависимостей. Исправление неоднозначных пользовательских определений — отдельная адресная операция, не автоматическое удаление или объединение.

## 5. Vector gate и финальные constraints

Helper проверяет обе коллекции и добавляет nullable `connection_id TEXT` с индексом только в существующие таблицы. **Data vectors:** catalog Table/Column ID доказывает владельца; CID, metadata и UUIDv5 backfill не вызывают модель и сохраняют embedding/content. Неизвестные IDs остаются с NULL CID; имя БД не используется для угадывания владельца. Противоречивые identities/unsupported schema блокируют транзакцию.

**Semantic vectors:** старый random-ID bulk text мог содержать samples нескольких источников. Его CID снимается, исходная строка сохраняется для аудита и не исполняется. После commit helper строит новые source-scoped Term/ColumnAttribute/SQL-definition projections с фильтрацией **до текста**. Уже канонические scoped projections переиспользуются. Исключение — подтверждённые `PqlAnalysis`: их изначально глобальные templates сохраняют embedding, получают отдельный global UUIDv5 и остаются с NULL CID; только явный global-PQL клиент может их читать.

`vectors` показывает состояние schema, pending updates, unresolved/conflict IDs. `semantic_rebuild_required` и `semantic_rebuild_failed` перечисляют CID. Ошибка/отсутствие embedding service после commit даёт exit 2, `applied: true`, `vector_migration_complete: false`, `ready_for_querying: false`: metadata уже применена, повторный `--apply` безопасно продолжает rebuild. Не откатывать metadata вручную ради повтора.

Отсутствующая коллекция не создаётся с угаданной размерностью. `sources_ready_for_querying` требует наличие всех ожидаемых data/semantic projections каждого источника; недостающие data vectors восстанавливаются штатным source-aware ingest с правильной embedding model. `vector_migration_complete` не заменяет этот gate: metadata migration может быть закончена при пустом/неполном vector index. Финальный `ready_for_querying` — только техническая проверка покрытия IDs, не замена smoke-тестов маршрутизации, разрешений и provenance.

Только после завершения согласованного backfill и проверки source-aware writers применить финальную revision:

```bash
PYTHON_DOTENV_DISABLED=1 uv run --no-sync alembic upgrade e38c06a17f42
PYTHON_DOTENV_DISABLED=1 uv run --no-sync alembic check
```

Цепочка включает constraints revision `d27b94f58c31` и retirement revision
`e38c06a17f42`: непустые уникальные ярлыки, повтор физических имён,
source-scoped уникальность SQL metadata и таблицу `retired_connection`.
Helm pin и runtime schema gate этого релиза ожидают `e38c06a17f42`.
Проверить одинаковые ID-наборы у API/ingestion, env-first при непустом
`CONNECTION_STRINGS`, выбранный source результата и отсутствие изменений
соседа после scoped reset. До этих проверок API/workers не активировать.

## 6. Rollback

После появления двух одинаковых физических имён старый name-based код небезопасен. Обычный downgrade не должен удалять или объединять строки ради старого unique constraint; при конфликте он завершится ошибкой с откатом транзакции.

Retirement revision также запрещает downgrade при непустом
`retired_connection`: удаление tombstones лишило бы систему запрета повторного
использования CID. Не очищать эту таблицу ради downgrade. Backup/restore
должен включать её вместе с catalog, vectors и frontend history.

Для rollback остановить всех writers и восстановить **согласованный** pre-upgrade backup catalog/vector/frontend с прежним комплектом сервисов либо выполнять source-aware forward fix. Восстановление старого backup не включает новые данные, записанные после него: решение об их сохранении принимает оператор. Legacy Vault refs остаются нетронутыми. Новые ID-paths чистить только адресно после проверки, что они больше не используются, с отдельным разрешением; массовое удаление Vault недопустимо.

### Локальная репетиция миграции и restore (5.2)

На task-owned disposable PostgreSQL 17 parent проверил новый opt-in
`dev_tools/tests/test_connection_identity_rehearsal.py`: **1 passed, 34.60 s**.
Настоящие `pg_dump -Fc` / `pg_restore --exit-on-error` завершились с кодом 0.
Backup сделан до additive upgrade и восстановлен в другую пустую БД; fingerprint
всех строк и column shapes схем `public`, `frontend`, `vdb` совпал с исходным.
БД с появившимися duplicates при этом не изменилась. Archive приватный (0600),
в безопасном receipt только hashes, counts и exit codes; две собственные БД
удалены обычным DROP без FORCE. Независимое review одобрено.

Проверены read-only check, явное разрешение origin ambiguity, два apply без drift,
неизменность IDs/descriptions/inline credentials и точного legacy reference,
NULL legacy chat/history, backfill data-vector identity без изменения embedding,
отсутствующая semantic collection и отказ readiness до открытия connector.
Downgrade после двух `sales` отклонён атомарно с сохранением head и обоих roots.
Restore вернул legacy revision и историю до добавления CID, а не смешал старую
историю с новым состоянием каталога.

Дополнительная parent-матрица из полного завершённого suite: **117 passed,
0 skipped** — schema migration (6), metadata helper (8), vectors (34), env/managed
sources (30), conversation binding (39). Она покрывает label/origin conflicts,
неопределённый SQL ownership, shared-Term scoped semantic rebuild, missing vectors,
NULL read-only chats, env-first переключение и восстановление managed IDs.
Вместе с новой репетицией: **118 passed**. Это не production rollout и не проверка
Vault: внешние источники и embedding endpoints для rehearsal не использовались.
Nullable frontend CID в backup-тесте добавлен SQL-моделью Prisma-поля; настоящий
Prisma schema push отдельно прошёл на изолированной build DB (20 таблиц), а не
на пользовательской БД. Опасные команды теста закрыты отдельным opt-in и guards
точного task-owned контейнера/loopback endpoint.

## Env-источники: CONNECTION_STRINGS

Непустой `CONNECTION_STRINGS` выбирает только env-набор во всех процессах; managed-корни остаются сохранёнными и возвращаются после снятия env-настройки. Пустая или пробельная строка означает managed-only режим. Ошибочный непустой набор не включает fallback.

Каждый URL может содержать зарезервированные query-поля `metadata_connection_id` (UUID) и `metadata_label` (непустой URL-encoded ярлык). Они удаляются до передачи URL драйверу; остальные query-options сохраняются. Пример двух синтетических источников с одинаковым физическим именем, без credentials:

```text
CONNECTION_STRINGS=postgresql://local.invalid/sales?metadata_connection_id=11111111-1111-4111-8111-111111111111&metadata_label=Sales%20local,postgresql://prod.invalid/sales?metadata_connection_id=22222222-2222-4222-8222-222222222222&metadata_label=Sales%20production
```

Одинаковые физические имена внутри набора требуют явных разных ID и явных уникальных ярлыков. Повтор ID/зарезервированного поля, неверный UUID, пустой ярлык, конфликт origin или нормализованного ярлыка отклоняет весь набор. PostgreSQL сравнивает ярлыки через `lower(TRIM(BOTH FROM display_label))`, включая неактивные корни; Python casefold не применяется.

Plain URL без ID сохраняет совместимость только при однозначности: новый свободный physical name получает сохранённый UUID; единственный существующий env-корень переиспользуется. Перестановка URL и ротация пароля ID не меняют. Managed/catalog-only корень нельзя присвоить при startup: сначала требуется явное решение origin через migration helper. После появления нескольких одноимённых корней (даже неактивных) bare-name больше не разрешается — укажите UUID. Новый явный env-ID рядом с managed-источником требует собственного свободного ярлыка. Существующий явный ID должен принадлежать env и той же физической БД; можно переименовать ярлык, но не переписать managed ownership или physical name.

API и ingestion регистрируют весь env-набор одной транзакцией с advisory lock после проверки версии схемы, до workers/schedulers и readiness. Повторный startup идемпотентен; credentials из env не сохраняются. Runtime discovery только читает metadata: незарегистрированный ID или несовпавшая конфигурация дают `source_not_ready`, а не скрытую регистрацию. Factory/OpenAPI не регистрируют источники. Этот metadata-этап не заменяет последующий перевод registry, credentials и query paths на source IDs и не означает готовность полного релиза.

## Managed lifecycle: credentials и адресные изменения

### Query preflight и discovery

`GET /api/connections/sources` возвращает только `connection_id`,
`display_label`, `database_name` и `dialect`. Discovery читает metadata без
Vault и открытия analyzed databases; отдельное право Next proxy `chat:use`
проверяется в клиентском пакете 4.4, settings API не заменяет этот маршрут.

Перед query/coverage resolver принимает ровно один `connection_id`,
`connection_label` либо legacy `target_db`. Явный ID сравнивается точно;
legacy физическое имя допустимо только при однозначности. Без селектора
единственный активный источник выбирается автоматически, несколько требуют
явного выбора. Неполная миграция/индексация выбранного источника возвращает
`source_not_ready` до model/SQL и history/analytics writes.

Минимальный runtime readiness gate требует source-bound semantic context и
строки выбранного CID в обеих VDB collections с CID в поисковых ключах
валидного полного индекса. NULL/global PQL и строки соседа не подтверждают
готовность. Gate только читает store: не создаёт таблицы, не вызывает модель
и не открывает source connector. Он **не заменяет** проверку полного набора
projection IDs migration helper и сквозной smoke test.

Принятый источник фиксируется в query state/retries и безопасном provenance:
первый SSE `source` и `answer.source` содержат один boundary snapshot. Vector
metadata не доказывает ownership node ID: candidate/model context повторно
подтверждается source-scoped DAL, а дополнительные Table-hits заменяются
persisted metadata. Model-authored value probes проверяют полные
catalog/schema/table identities до SQL и cache feedback; совпадения короткого
имени недостаточно. Conversation binding проверяется отдельным атомарным
протоколом ниже, а не самим query resolver.

Проверка backend/dev-tools после query-пакета (2026-11-08): **3143 passed,
146 skipped**, Ruff lint/format (**519 files**), SPDX и OpenSpec strict —
green. Среди пропусков **125 golden checks**: Pagila/Chinook fixture ещё не
seeded; это не доказанный golden replay. Для prediction-тестов установлены
только 9 пакетов существующего frozen Kumo extra, без изменения lockfile и
удаления прочих пакетов. Проверки MCP/frontend, golden regeneration/replay,
сквозной acceptance и rollout по-прежнему не завершены.
Финальная проверка дополнительно исключает semantic FK target из query
hydration, если канонический `table_id`/`source_column` атрибута не совпадает
с owning колонкой, даже когда сама колонка принадлежит выбранному CID.

### Conversation binding и manual append

Вопрос с `conversation_id` сначала проходит owner check и row lock в общей
транзакции. Пустой unbound диалог получает CID вместе с первым принятым
сообщением; bound без селектора наследует сохранённый CID. Явный другой
активный CID возвращает `409 source_conflict` до проверки его query readiness.
NULL CID при наличии любого сообщения означает
`409 legacy_conversation_read_only`, независимо от нового селектора.
Отклонённый запрос не меняет binding, сообщения, analytics или timestamp.

Manual `POST /api/conversations/{id}/messages` сохраняет право
`conversation:write`, прежний JSON message shape и статус 201. Next передаёт
его backend counterpart `POST /api/chat/conversations/{id}/messages` без
Prisma insert и без Bearer/SSO. Counterpart использует тот же DAL transaction
protocol, проверяет active metadata, не запускает model/connector и не создаёт
отсутствующий диалог. Missing/foreign conversation дают одинаковый 404.
Селекторы ID/label/legacy mutually exclusive; bound можно продолжать без них.

SSO и коннектор проверяются после принятия binding, вне row lock. Отказ
выбранного активного источника после принятия turn остаётся его обычной
ошибкой; missing SSO token сохраняет HTTP 401 без PAT fallback. Поздние
assistant/chart/error writes проверяют expected CID, owner и active metadata
под блокировкой диалога. Отказ этой проверки не запускает повторную запись
ошибки и не перепривязывает историю. Проверка активности — snapshot, не
общая сериализация с concurrent source deletion.

DB race-проверки наблюдают `pg_blocking_pids` у двух независимых транзакций,
а не полагаются на sleeps: разные CID дают одного победителя, одинаковый CID
сериализуется. Дополнительные HTTP→disposable-PostgreSQL проверки проходят
через настоящие DAL и router; model/worker/SSO — deterministic stubs.

Итог после binding/manual пакета: **3212 passed, 146 skipped** в полном
backend/dev-tools наборе; Ruff (**524 files**), SPDX и strict validation —
green. Manual proxy/shared opt-in source-header проверены **10 Vitest tests**,
scoped ESLint/Prettier и `tsc --noEmit` проходят. Backend/public/bundled MCP
OpenAPI regeneration повторён без drift. Это не завершает MCP implementation,
UI acceptance или golden replay; их release gates остаются отдельными.

### MCP и optional semantic projections

`list_sources` использует отдельный `GET /api/connections/sources` с `chat:use`,
а не settings-only `/connections/source`. Discovery не открывает source drivers
и не возвращает credentials. `ask_question` принимает взаимоисключающие ID/label,
сохраняет отсутствие селектора для inherited conversation и возвращает безопасный
typed `source`. Все присутствующие SSE/result snapshots должны совпадать;
malformed/mixed provenance отвергается без raw payload. Старый backend без обоих
полей остаётся совместимым через `source: null`, без выдуманного источника.

Curated semantic reads принимают optional ID/label/legacy selector: передают CID
в counts, certification, related traversal, column paths, SQL и table detail до
pagination/hydration. Без selector остаётся global glossary overview, не auto-pick
исполняемого источника. Неизвестный/чужой table или SQL attribute имеет тот же 404,
что отсутствующий. Coverage Next proxy передаёт только явно разрешённый caller
SSO token; общий proxy по умолчанию продолжает отбрасывать Bearer.

Проверка пакета 4.3: **3250 passed, 146 skipped** backend/dev-tools; **188 passed**
MCP, включая 28 generated-tool serialization сценариев; **23 passed** Vitest.
Ruff, scoped frontend ESLint/Prettier, TypeScript, SPDX и OpenSpec strict — green.
Backend/public/bundled OpenAPI повторно генерируются без drift; независимые ревью
MCP/Next, terms/SQL и table detail одобрены. UI acceptance, golden replay и rollout
этими результатами не подтверждаются.

### Connection и catalog UI

Connection cards и все mutations/test/SSO используют CID. Label-only save
отправляет только `display_label`, не требует test/полных driver fields и
перечитывает список; physical name остаётся параметром драйвера. Catalog API
добавляет `display_label` отдельно от physical `name`; старые ID cache keys
сохранены. Новый ярлык предлагается по physical name, только если свободен
именно ярлык; ручной ввод не перезаписывается.

Проверка 4.4: **3254 passed, 146 skipped** backend/dev-tools; **38 passed** Vitest,
TypeScript/scoped ESLint/Prettier/Ruff/OpenSpec — green. Отдельный реальный
PostgreSQL прогон catalog label/readback — **4 passed**. Независимые ревью
catalog/proxy и connection UI одобрены. В браузере на реальных React components
с синтетической API boundary проверены две карточки `sales`, rename только A,
неизменность B, один label-only PUT без test POST, нормализованный конфликт,
автоподстановка свободного ярлыка и сохранение ручного ярлыка после смены БД.
Harness не читает `.env` и не ходит в настоящие источники; это функциональный
check, не проверка production auth/полного визуального оформления. Public/bundled
OpenAPI повторно сгенерированы без drift. Chat UI и полный release acceptance
остаются следующими gates.

### Повторная проверка query helper contracts и real-graph regression

Полный graph в 5.1 обнаружил ранее не покрытый `TypeError`: caller передавал CID
в junction helper без такого параметра. Pipeline подавлял ошибку и доходил до SQL
generation без schema context. Parent воспроизвёл RED на настоящем графе; 4.1 была
повторно открыта. Sweep обнаружил ещё пять несовместимых DAL helpers. Исправлены
общие attributes/exploration paths, а не убран CID у callers: ownership обоих
концов FK, canonical table/source-column и source filters применены до traversal,
count и hydration. Независимый focused прогон — **45 passed**, review одобрен.
Повторный статический audit: **309 resolved CID calls, 0 несовместимых**; ещё
**50 dynamic/method calls** этим audit не разрешены и не объявляются проверенными.

Parent opt-in `test_same_name_connection_lifecycle.py`: **2 passed, 30.23 s**.
Два отдельных disposable PostgreSQL содержат одинаковые `sales/public.orders`,
разные настоящие SQL markers и `draft_status` только у A. Реальны ingest, pgvector,
coverage/query graphs, worker loop, conversation persistence, HTTP/SSE и stateless
MCP через отдельный venv. Подменены только model/embedding computation и worker
scheduling. Проверены rename/follow-up, sole-source workflow, reset/re-ingest A,
неизменность catalog/semantic/SQL/vector snapshots B, retirement/tombstone и
сохранение недоступной для продолжения истории. Negative oracles обнаруживают
снятый CID filter по raw foreign hit и физическую подмену коннектора по marker;
подлинный несовпадающий CID отклоняется до модели. Независимое QA review одобрено.

Границы: semantic Term/attributes заданы fixture вручную, автоматическая semantic
compilation не проверяется. Retirement env-root использует явно разрешённый
fixture-only detach после исключения из env и доказанной неактивности; managed-only
guard не ослаблен, finalizer настоящий. Private HTTP fixture не проверяет gateway
auth, spawn/cancel или полный HTTP delete. Известный старый zero-distance coverage
edge case (`score or inf`) не исправлен: fixture использует небольшой положительный
cosine distance. Повторный полный backend после helper fixes: **3299 passed,
148 skipped, exit 0** (602.70 s); из skips два — явный opt-in E2E, отдельно
прошедший выше. DAL surface обновлён, Ruff lint/format — green. 4.1 повторно
принята, 5.1 закрыта после parent execution и обоих независимых reviews.

Изолированная frontend build без настоящих `.env`/npm credentials прошла с
**exit 0**, compile/TypeScript/prerender **69/69**. Использованы неизменённые
507 frontend-файлов, настоящая disposable Prisma DB (20 таблиц) и synthetic
loopback backend только для build-time reads. Неизвестных HTTP запросов нет;
собственные DB и fixture server очищены. Это проверка сборки, не auth acceptance.

### Golden fidelity и gateway acceptance

Golden fixture создана в отдельном новом PostgreSQL/pgvector контейнере и
временном Chinook SQLite. Старый baseline: **26 failed, 102 passed, 0 skipped**.
Один capture caller не передавал обязательный CID SQL-doc helper; поведенческий
контрактный тест воспроизвёл RED, минимальная передача CID дала **6 passed**.
Это исправление capture, а не ослабление source guard в DAL.

Обновлённый эталон: **121 read, errors={}**. Source-only projections
`[pagila|chinook]` заменили три прежних global/zone captures. Parent проверил все
**119 общих reads**: прежние значения и list members сохранены, изменения только
аддитивные (CID, label, source embedding context; ранее optional `owner_id: null`).
Три независимых captures совпали побайтово; raw/unmapped UUID отсутствуют.
После независимого review эталон записан штатным генератором. Parent replay
**tracked** golden: **127 passed, 0 skipped, 4.10 s**, exit 0. Это отдельный
реальный replay, не трактовка default skipped checks как успеха.

На неизменённой production build запущен отдельный разрешённый local gateway с
настоящими Better Auth sessions, Prisma DB и ролями; backend заменён private
synthetic boundary, не production. Parent независимо повторил **12 assertions**:
anonymous source discovery → 401; viewer получает два безопасных source descriptors;
его PUT/DELETE/SSO mutations → 403 без upstream I/O; чужой thread → 404; owner
читает bound CID и legacy NULL history. В подготовительной HTTP-матрице дополнительно
проверены label-only admin PUT и невозможность rebind через create/title PATCH
(17 checks). Browser cookies пользователя не менялись; parent HTTP использовал
свои изолированные test sessions, не настоящие credentials.

Parent дополнительно вручную выполнил MCP protocol acceptance через настоящий
loopback HTTP/SSE/full worker graph, с отдельным MCP venv: safe two-`sales`
discovery, explicit ID A и label B вернули разные реальные SQL markers,
multi-source без выбора отклонён до модели, sole-source без выбора использовал B.
Никаких unexpected model calls; временный store закрыт обычным DROP.

### Финальные локальные quality gates (5.3)

- Последний полный backend/dev-tools прогон после всех изменений: **3299 passed,
  148 skipped, exit 0, 476.75 s**. В стандартном store нет golden fixture;
  настоящий golden replay (**127/127**) и отдельные opt-in E2E (**2/2**) / restore
  rehearsal (**1/1**) прошли отдельно, skips не объявлены выполненными проверками.
- MCP **188 passed**; frontend **87 Vitest passed**, TypeScript, полный Prettier и
  ESLint — exit 0. Единственное ESLint warning — прежний `common/UserMenu.tsx:42`.
  Полная изолированная build прошла; fingerprint 507 frontend-файлов остаётся тем же.
- Ruff check и format-check **552 files**, SPDX/контрактные проверки, OpenSpec
  **1.16.1** strict, `git diff --check` — exit 0. Backend и public/bundled OpenAPI
  повторно генерируются без drift (64 paths/88 ops; 84 paths/123 ops).
- `crapper`, `dryer`, `mutator` недоступны и не устанавливались: CRAP < 10,
  duplication threshold 0.8 и полный mutation score **не измерены**. Два целевых
  mutation-oracles E2E проверены, но не заменяют полный mutator.

Remote CI/PR не запускались: commit, push и открытие PR не запрашивались. Git index
сохранён; локальные gates не означают разрешения production rollout или Vault cleanup.

### Локальная acceptance завершена (5.4)

Итоговые независимые code review и QA evidence review: **APPROVE**, подтверждённых
блокирующих замечаний нет. Сверены обе delta-spec, query/lifecycle/credential
boundaries, реальные A/B control answers, migration/restore receipts, UI/MCP
acceptance, права и ownership. Все 21 implementation-задачи приняты локально;
перечисленные выше ограничения не выданы за проведённые проверки.

Новые source/golden контейнеры и эксклюзивные anonymous volumes очищены только
по ownership receipts после проверки отсутствия clients. Свои build/gateway БД
удалены обычным DROP, gateway/backend fixture и Vite harness остановлены;
существующий metadata-контейнер не удалялся. Временный synthetic login file удалён.
Browser MCP поставлен пользователем на паузу при попытке закрыть тестовую вкладку:
повторных вызовов не было, вкладка не закрыта автоматически. Это не отменяет ранее
полученные browser results и не требует возобновления для завершения реализации.

Изменения остаются незакоммиченными, пользовательский Git index не менялся.
Архивация change, PR/CI, production activation и Vault cleanup — отдельные действия,
а не выполненные части этой локальной acceptance.

### Chat UI binding и recovery

Conversation selects/API возвращают nullable CID; create/title PATCH по-прежнему
не записывают его. UI использует отдельный `chat:use` discovery, не settings API.
Новый multi-source чат ждёт явного выбора, single-source допускает sole-source
auto-select. Bound thread имеет locked source; legacy/deleted-source история
доступна read-only. Discovery outage сохраняет binding и историю и предлагает
retry, не означает «источник удалён» и не выбирает соседний источник.

После HTTP refusal UI перечитывает detail: SSO-отказ может следовать за принятием
turn. Неудачный recovery блокирует selector/send до успешного явного retry, без
auto-resend. Source events/results проверяются на CID и безопасную форму;
поздние callbacks не меняют другой thread. Глобальный semantic-ready status не
скрывает историю: selected-source readiness остаётся backend gate.

Parent-проверка 4.5: **87 Vitest passed**, TypeScript, scoped ESLint/Prettier,
**188 MCP passed**, SPDX/OpenSpec strict — green. Public/bundled spec stable,
Git index hash не менялся. Независимые transport и UI/Prisma ревью одобрены.
В браузере на реальных React components/hook с synthetic API проверены 0/1/2
источника, selected CID в first/follow-up, locked header, reload bound chat,
legacy/deleted read-only, discovery retry и SSO-after-acceptance recovery.
Удержанные detail/create/SSE responses выпущены после переключения в legacy:
CID, история и URL нового выбора не изменились; запоздалый create не вызвал chat
POST. Console errors в финальном race-сценарии отсутствуют. Это не заменяет
integrated source-DB/pgvector regression, migration rehearsal или полный build.

### Ingestion и semantic compilation

Оба schedulers получают один metadata-only env-first список CID; одинаковые
physical names не объединяются. Каждый коннектор открывается внутри обработки
своего CID: отказ соседнего источника не меняет выбранный ID. Internal
`POST /ingest` принимает только `{ "connection_id": "..." }`, не credentials;
неизвестный/неактивный ID отклоняется до queue/driver/model calls.

Для адресной developer compilation используется
`uv run --no-sync python -m auto_ontology.semantic --connection-id <connection_id>`.
CLI проверяет схему и регистрирует env metadata до compilation. Sampling,
breaker и SQL/FK/bridge-контекст ограничены этим CID. Summary читается из
`.semantic_summaries/<URL-encoded-connection-id>.json`; старого fallback на
physical name нет. Подтверждённый summary переносится оператором для каждого
источника явно; ярлык в файловом ключе не используется. PII propagation после
прохода изменяет только атрибуты выбранного CID. Глобальный compile-trigger
по-прежнему означает явный проход по всему активному набору, не выбор первого.

### Credentials и изменения

Сервис управления выбирает существующий managed-корень по `connection_id`; legacy physical-name adapters допускаются только при единственном managed-совпадении. Они читают metadata, а не перебирают все credentials. Env-корни и catalog-only корни через UI не изменяются. При непустом `CONNECTION_STRINGS` изменения managed-подключений блокируются: сначала оператор явно возвращает managed-режим; неактивные записи не удаляются и не переписываются.

Новый источник получает серверный UUID **до** Vault write и собственный путь `connections/<connection_id>`. Даже при одинаковом physical name он не читает и не перезаписывает legacy secret. Проверка ярлыка выполняется до записи секрета через PostgreSQL `lower(TRIM(BOTH FROM ...))`; окончательную уникальность обеспечивает DB constraint, в том числе при конкурентном создании. Явный пустой ярлык отклоняется. Без ярлыка используется physical name только если он свободен, без автоматических суффиксов.

Существующий `credential_ref` остаётся точным прежним путём при чтении, редактировании, SSO toggle и удалении. Если Vault для этого ref не настроен, секрет отсутствует либо чтение/запись не удались, операция завершается `source_unavailable`: inline JSON, одноимённый legacy path и соседний источник не используются как fallback. Inline credentials остаются на своей строке, когда Vault выключен; явное сохранение connection settings при включённом Vault может перенести **этот же ID** на его собственный новый ID-path. SSO toggle сохраняет текущий inline/ref способ хранения и не запускает ingestion.

Пустые secret-поля при edit/test сохраняют значения только выбранного ID. Physical database name и driver type неизменяемы; endpoint остаётся явной настройкой оператора без новой политики доказательства тождественности сервера. Label-only edit меняет только metadata: без чтения credentials, сетевой проверки, worker/model refresh и re-ingest. Изменение schema/table filters по-прежнему требует ingestion.

Если Vault write нового источника успешен, но DB insert не состоялся (включая конкурентный конфликт ярлыка), сервис делает единственную компенсацию: удаляет только новый ID-path этой попытки. Legacy references и секрет победившего источника не затрагиваются. При отказе компенсации активная запись не появляется; безопасный журнал содержит `orphan_secret` и новый UUID без секретов, URL и исходного текста исключения. Оператор проверяет этот UUID и отсутствие активной привязки, затем отдельно согласует адресную очистку. Автоматической массовой очистки нет; Vault delete helper вообще не принимает пустой путь как «удалить всё».

Удаление сначала удаляет только точный ref (если он есть), затем очищает
`configuration_origin`, `connection` и `credential_ref` только выбранного ID.
Задание cleanup получает захваченный ID, не повторно разрешённое имя.
Окончательная очистка отделена от обычного reset; HTTP 202 подтверждает только
постановку задания. Полный пакет query/history/UI ещё должен пройти общие
release gates — успешный lifecycle-запрос не разрешает частичный rollout.

### Reset, final remove и повтор cleanup

Обычный scoped/global reset сохраняет root identity и её origin, label,
credentials/ref; очищаются принадлежащие выбранным CID слои анализа. Общий
Term не удаляется, пока нужен другому источнику. Global PQL vocabulary и его
embeddings сохраняются и при reset-all: это не source-owned данные. Записи
с недоказанным NULL ownership не назначаются выбранному источнику по имени.
Раздельный semantic reset сохраняет ingestion SQL history для rebuild;
data-only reset сохраняет SQL оставшихся semantic definitions. После удаления
links/owners удаляется только SQL выбранного CID без оставшихся catalog или
semantic references. Combined reset очищает оба слоя; глобального SQL sweep нет.

Final remove адресован только detached CID. После source-scoped metadata и
vector cleanup finalizer под registration lock повторно проверяет retirement,
записывает tombstone и удаляет этот root одной транзакцией. Зарегистрированный
активный root finalizer не удаляет. Ярлык освобождается для **нового** CID;
старый CID запрещён для env/bootstrap, managed insert, model import и stale
root upsert. История диалогов не удаляется и не перепривязывается.

Если cleanup завершился ошибкой, повторить разрешённое оператором задание с
**тем же захваченным CID**, даже если root уже отсутствует. Не разрешать его
по прежнему ярлыку: ярлык уже может принадлежать новому подключению. Internal
`POST /ingest/delete?connection_id=...` не требует активного источника именно
для такого повтора; ошибка задачи проверяется отдельно от HTTP 202. Legacy
`database_name` отклоняется, а не превращается в глобальную очистку.

Semantic reset принимает optional `connection_id`; отсутствие означает явный
reset-all. Режим pause → abort → drain → delete → resume → rebuild и общий
reset-lock сохранены. Tombstones обычными resets не очищаются. Они защищают
application writers; ручной SQL администратора обходит DAL и требует отдельной
ответственности. Миграция не выдумывает неизвестные исторически удалённые IDs.
Если compilation не завершилась до drain timeout, reset отменяется без delete
и rebuild; scheduler обязательно снимается с pause. Дождаться завершения pass
перед адресным повтором, не считать прежний HTTP 202 подтверждением очистки.

## Проверки metadata-этапа

На одноразовом PostgreSQL 17/pgvector проверены: additive → backfill → final upgrade с сохранением IDs/описаний, отклонение normalized label conflicts и небезопасного downgrade, read-only check, повторный apply, отсутствие частичной записи при конфликтах и явное задание legacy reference. Автоматические тесты: `auto_ontology/dal/tests/test_connection_identity_schema.py` и `dev_tools/tests/test_connection_identity_migration.py`.

Vector tests (`dev_tools/tests/test_connection_identity_vectors.py`) дополнительно проверяют JSON/JSONB metadata, model-free data backfill, quarantine mixed semantic text, scoped deterministic rebuild, повтор без model calls и retryable failure receipts. Live tests включаются только с `CONNECTION_IDENTITY_MIGRATION_TEST=1`, отключённым dotenv и явно заданным localhost-store. Каждый тест создаёт собственную временную БД и удаляет только её; номер порта не зафиксирован в тестах. Эти проверки не являются production rehearsal и не доказывают готовность полного query/UI релиза.

Проверка пакета до reset/delete включительно (2026-11-08): **1630 passed,
10 skipped**, Ruff lint/format — green, OpenSpec strict — green. Проверены
source-scoped reset на реальном pgvector, сохранность roots/соседа/global PQL,
повтор cleanup, tombstone writer guards (включая list/tuple labels), rollback и
конкурентная регистрация; новый head и Helm pin совпадают. Отдельные проверки
подтверждают сохранение SQL history/definitions при раздельном reset и отмену
reset без delete/rebuild при drain timeout. Это целевой regression-набор,
**не полный backend/MCP/frontend suite и не разрешение на rollout**.
