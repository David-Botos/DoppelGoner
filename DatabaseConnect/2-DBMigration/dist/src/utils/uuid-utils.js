"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IdConverter = void 0;
// Utility class for ID conversion
class IdConverter {
    constructor(namespace = "migration") {
        this.namespace = namespace;
        this.idCache = new Map();
    }
    convertToUuid(originalId) {
        if (!originalId)
            return null;
        // If already in UUID format, return as is
        if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(originalId)) {
            return originalId;
        }
        // Check cache first
        if (this.idCache.has(originalId)) {
            return this.idCache.get(originalId);
        }
        // Generate deterministic UUID from original ID
        const uuid = this.generateUuidFromString(`${this.namespace}:${originalId}`);
        this.idCache.set(originalId, uuid);
        return uuid;
    }
    generateUuidFromString(input) {
        // Implementation of UUID v5 (name-based)
        // Note: In a real implementation, use a proper UUID library
        const crypto = require("crypto");
        const hash = crypto.createHash("md5").update(input).digest("hex");
        return [
            hash.substring(0, 8),
            hash.substring(8, 4),
            "5" + hash.substring(13, 3), // Version 5
            "8" + hash.substring(17, 3), // Variant 8
            hash.substring(20, 12),
        ].join("-");
    }
}
exports.IdConverter = IdConverter;
