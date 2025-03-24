/RDSMigration/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts                 # Main entry point
│   ├── config/
│   │   ├── config.ts            # Configuration for connections and settings
│   │   └── migration-order.ts   # Defines migration order and dependencies
│   ├── connectors/
│   │   ├── snowflake.ts         # Snowflake connection handler
│   │   └── postgres.ts          # PostgreSQL connection handler
│   ├── models/
│   │   ├── common.ts            # Shared interfaces
│   │   ├── organization.ts      # Organization model definitions
│   │   ├── service.ts           # Service model definitions
│   │   ├── location.ts          # Location model definitions
│   │   ├── service-at-location.ts # Service at location model
│   │   ├── address.ts           # Physical and postal address models
│   │   └── phone.ts             # Phone model
│   ├── transformers/
│   │   ├── organization.ts      # Organization data transformer
│   │   ├── service.ts           # Service data transformer
│   │   ├── location.ts          # Location data transformer
│   │   ├── service-at-location.ts # Service at location transformer
│   │   ├── address.ts           # Address transformer
│   │   └── phone.ts             # Phone transformer
│   ├── loaders/
│   │   ├── organization.ts      # Organization PostgreSQL loader
│   │   ├── service.ts           # Service PostgreSQL loader
│   │   ├── location.ts          # Location PostgreSQL loader
│   │   ├── service-at-location.ts # Service at location loader
│   │   ├── address.ts           # Address loader
│   │   └── phone.ts             # Phone loader
│   ├── extractors/
│   │   ├── organization.ts      # Organization Snowflake extractor
│   │   ├── service.ts           # Service Snowflake extractor
│   │   ├── location.ts          # Location Snowflake extractor
│   │   ├── service-at-location.ts # Service at location extractor
│   │   ├── address.ts           # Address extractor
│   │   └── phone.ts             # Phone extractor
│   ├── utils/
│   │   ├── logger.ts            # Logging utility
│   │   ├── validation.ts        # Validation helpers
│   │   └── error-handler.ts     # Error handling and reporting
│   └── services/
│       ├── migration-service.ts # Controls migration flow
│       └── migration-logger.ts  # Logs migration progress to DB
└── scripts/
    ├── setup-postgres.sql       # PostgreSQL schema setup
    └── validate-migration.ts    # Post-migration validation