"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrganizationTransformer = void 0;
const transformer_1 = require("./transformer");
const uuid_1 = require("uuid");
class OrganizationTransformer extends transformer_1.Transformer {
    constructor(idConverter) {
        super(idConverter);
    }
    transformSingleRecord(source, translation) {
        const newId = (0, uuid_1.v4)();
        return {
            id: newId,
            name: source.NAME,
            alternate_name: source.ALTERNATE_NAME || undefined,
            description: translation?.DESCRIPTION || undefined,
            email: source.EMAIL || undefined,
            url: source.WEBSITE || undefined,
            year_incorporated: source.YEAR_INCORPORATED || undefined,
            legal_status: source.LEGAL_STATUS || undefined,
            parent_organization_id: this.idConverter.convertToUuid(source.PARENT_ORGANIZATION_ID),
            last_modified: new Date(source.LAST_MODIFIED).toISOString(),
            created: new Date(source.CREATED).toISOString(),
            original_id: this.idConverter.convertToUuid(source.ID) || source.ID.toString(),
            original_translations_id: translation?.id !== undefined
                ? this.idConverter.convertToUuid(translation.id)
                : null,
        };
    }
}
exports.OrganizationTransformer = OrganizationTransformer;
