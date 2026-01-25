/* Dashtable - High-performance segmented hash table */

#ifndef DASHTABLE_H
#define DASHTABLE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Number of slots per bucket
 */
#define dash_SLOTS_PER_BUCKET 14

/**
 * Number of regular buckets per segment
 */
#define dash_REGULAR_BUCKETS 56

/**
 * Number of stash buckets per segment
 */
#define dash_STASH_BUCKETS 4

/**
 * Total buckets per segment
 */
#define dash_TOTAL_BUCKETS (dash_REGULAR_BUCKETS + dash_STASH_BUCKETS)

/**
 * Maximum entries per segment
 */
#define dash_SEGMENT_CAPACITY (dash_TOTAL_BUCKETS * dash_SLOTS_PER_BUCKET)

/**
 * Cache line size (64 bytes on most modern CPUs)
 */
#define dash_CACHE_LINE_SIZE 64

/**
 * Opaque iterator handle
 */
typedef struct dash_DashIterHandle {
    uint8_t _private[0];
} dash_DashIterHandle;

/**
 * Opaque table handle
 */
typedef struct dash_DashTableHandle {
    uint8_t _private[0];
} dash_DashTableHandle;

/**
 * Table configuration
 */
typedef struct dash_DashConfig {
    uint32_t initial_segments;
    float max_load_factor;
} dash_DashConfig;

/**
 * Create iterator
 */
struct dash_DashIterHandle *dash_iter_new(const struct dash_DashTableHandle *table);

/**
 * Free iterator
 */
void dash_iter_free(struct dash_DashIterHandle *iter);

/**
 * Create a new table
 */
struct dash_DashTableHandle *dash_table_new(const struct dash_DashConfig *config);

/**
 * Free a table
 */
void dash_table_free(struct dash_DashTableHandle *table);

#endif  /* DASHTABLE_H */
