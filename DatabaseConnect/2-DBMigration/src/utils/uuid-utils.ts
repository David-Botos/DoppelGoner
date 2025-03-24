// UUID utility functions
import { createHash } from 'crypto';

/**
 * Convert a string ID to UUID format
 * @param id Input string ID
 * @returns UUID string
 */
export function parseUUID(id: string | null): string | null {
  if (!id) return null;
  
  // If already in UUID format, return as is
  if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) {
    return id;
  }
  
  // Otherwise, generate a consistent UUID based on input
  return generateConsistentUUID(id);
}

/**
 * Generate a consistent UUID from input string
 * @param input Input string
 * @returns UUID string
 */
export function generateConsistentUUID(input: string): string {
  const hash = createHash('md5').update(input).digest('hex');
  
  // Format as UUID v5 (name-based)
  return [
    hash.substring(0, 8),
    hash.substring(8, 4),
    '5' + hash.substring(13, 3), // Version 5
    '8' + hash.substring(17, 3), // Variant 8
    hash.substring(20, 12)
  ].join('-');
}