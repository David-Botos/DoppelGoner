"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Transformer = void 0;
class Transformer {
    constructor(idConverter) {
        this.idConverter = idConverter;
    }
    async transform(dataMap) {
        const transformedRecords = [];
        // Iterate through each entry in the data map
        for (const [id, { main, translations }] of dataMap.entries()) {
            // Simply use the first translation if available
            const translation = translations.length > 0 ? translations[0] : null;
            // Transform the record and ensure we resolve any promise
            const transformedRecord = await Promise.resolve(this.transformSingleRecord(main, translation));
            // Now transformedRecord is guaranteed to be of type R, not Promise<R>
            transformedRecords.push(transformedRecord);
        }
        return transformedRecords;
    }
}
exports.Transformer = Transformer;
