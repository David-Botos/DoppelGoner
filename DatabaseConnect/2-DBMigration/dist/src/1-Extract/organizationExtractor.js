"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrganizationExtractor = void 0;
const extractor_1 = require("./extractor");
class OrganizationExtractor extends extractor_1.Extractor {
    constructor(snowflakeClient) {
        super(snowflakeClient, {
            main: "ORGANIZATION",
            translations: "ORGANIZATION_TRANSLATIONS",
        });
        this.snowflakeClient = snowflakeClient;
    }
    async extractMainRecords(limit, offset) {
        const query = `
      SELECT 
        ID,
        NAME,
        ALTERNATE_NAME,
        EMAIL,
        WEBSITE,
        YEAR_INCORPORATED,
        LEGAL_STATUS,
        PARENT_ORGANIZATION_ID,
        LAST_MODIFIED,
        CREATED
      FROM ${this.sourceTables.main}
      ORDER BY CREATED DESC
      LIMIT ${limit}
      OFFSET ${offset}
    `;
        return this.snowflakeClient.query(query);
    }
    async extractTranslationRecords(ids, locale) {
        const formattedIds = ids.map((id) => `'${id}'`).join(", ");
        console.log("fetching translations records for ", formattedIds.slice(0, 100), "...");
        const query = `
      SELECT 
        ID,
        ORGANIZATION_ID,
        LOCALE,
        DESCRIPTION,
        IS_CANONICAL,
        ORGANIZATION_ID as PARENT_RECORD_ID
      FROM ${this.sourceTables.translations}
      WHERE ORGANIZATION_ID IN (${formattedIds})
      AND LOCALE = '${locale}'
    `;
        return this.snowflakeClient.query(query);
    }
}
exports.OrganizationExtractor = OrganizationExtractor;
