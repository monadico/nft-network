const fs = require('fs');
const path = require('path');
const { HoldersDatabase, processCollection } = require('./fetch-holders-db');
require('dotenv').config();

// Configuration
const TRENDING_COLLECTIONS_PATH = '../nft-backend/trending_collections.json';
const BATCH_SIZE = 5;
const DELAY_BETWEEN_BATCHES = 10000; // 10 seconds

class BatchProcessor {
    constructor() {
        this.db = new HoldersDatabase();
        this.processed = 0;
        this.successful = 0;
        this.failed = 0;
        this.errors = [];
        this.startTime = new Date();
    }

    async init() {
        await this.db.init();
    }

    loadTrendingCollections() {
        const fullPath = path.resolve(TRENDING_COLLECTIONS_PATH);
        
        if (!fs.existsSync(fullPath)) {
            throw new Error(`Trending collections file not found: ${fullPath}`);
        }

        const data = JSON.parse(fs.readFileSync(fullPath, 'utf8'));
        const collections = data.collections || [];
        
        console.log(`📚 Loaded ${collections.length} collections from trending_collections.json`);
        
        // Filter valid collections (must have id)
        const validCollections = collections.filter(c => c.id && c.id.startsWith('0x'));
        console.log(`✅ Found ${validCollections.length} valid collections with contract addresses`);
        
        return validCollections;
    }

    async filterCollectionsToProcess(collections, forceReprocess = false) {
        if (forceReprocess) {
            console.log(`🔄 Force reprocess mode: will process all ${collections.length} collections`);
            return collections;
        }

        const toProcess = [];
        
        for (const collection of collections) {
            const existing = await this.db.getCollectionInfo(collection.id);
            if (!existing) {
                toProcess.push(collection);
            }
        }

        const alreadyProcessed = collections.length - toProcess.length;
        console.log(`🔍 Collections already processed: ${alreadyProcessed}`);
        console.log(`📝 Collections to process: ${toProcess.length}`);
        
        return toProcess;
    }

    logSuccess(collection, result) {
        this.processed++;
        this.successful++;
        
        const elapsed = (new Date() - this.startTime) / 1000;
        const rate = this.processed / elapsed * 60; // per minute
        
        console.log(`✅ [${this.processed}] ${collection.name || collection.id.slice(0, 10) + '...'}`);
        console.log(`   Holders: ${result.ownerCount} | Rate: ${rate.toFixed(1)} collections/min`);
    }

    logFailure(collection, error) {
        this.processed++;
        this.failed++;
        this.errors.push({ collection: collection.id, error: error.message });
        
        console.log(`❌ [${this.processed}] ${collection.name || collection.id.slice(0, 10) + '...'}`);
        console.log(`   Error: ${error.message.slice(0, 100)}`);
    }

    async processCollectionBatch(collections, force = false) {
        for (const collection of collections) {
            try {
                const result = await processCollection(
                    this.db, 
                    collection.id, 
                    collection.name,
                    force
                );
                
                this.logSuccess(collection, result);
                
            } catch (error) {
                this.logFailure(collection, error);
            }
        }
    }

    logSummary() {
        const elapsed = (new Date() - this.startTime) / 1000;
        const minutes = Math.floor(elapsed / 60);
        const seconds = Math.floor(elapsed % 60);
        
        console.log(`\n📊 BATCH PROCESSING COMPLETE`);
        console.log(`⏱️  Total time: ${minutes}m ${seconds}s`);
        console.log(`✅ Successful: ${this.successful}`);
        console.log(`❌ Failed: ${this.failed}`);
        
        if (this.processed > 0) {
            console.log(`📈 Success rate: ${((this.successful / this.processed) * 100).toFixed(1)}%`);
        }
        
        if (this.errors.length > 0) {
            console.log(`\n❌ Failed collections:`);
            this.errors.slice(0, 10).forEach((err, i) => {
                console.log(`  ${i + 1}. ${err.collection.slice(0, 20)}... - ${err.error.slice(0, 50)}...`);
            });
            
            if (this.errors.length > 10) {
                console.log(`  ... and ${this.errors.length - 10} more failures`);
            }
        }
    }

    async run(options = {}) {
        const {
            force = false,
            limit = null,
            batchSize = BATCH_SIZE
        } = options;

        try {
            // Load and filter collections
            const allCollections = this.loadTrendingCollections();
            let collectionsToProcess = await this.filterCollectionsToProcess(allCollections, force);
            
            if (limit) {
                collectionsToProcess = collectionsToProcess.slice(0, limit);
                console.log(`🎯 Limited to first ${collectionsToProcess.length} collections for testing`);
            }

            if (collectionsToProcess.length === 0) {
                console.log('✅ All collections already processed! Use --force to reprocess.');
                return;
            }

            console.log(`\n🚀 Starting batch processing of ${collectionsToProcess.length} collections...`);
            console.log(`⚙️  Batch size: ${batchSize}`);

            // Process in batches
            for (let i = 0; i < collectionsToProcess.length; i += batchSize) {
                const batch = collectionsToProcess.slice(i, i + batchSize);
                const batchNum = Math.floor(i / batchSize) + 1;
                const totalBatches = Math.ceil(collectionsToProcess.length / batchSize);

                console.log(`\n📦 Processing batch ${batchNum}/${totalBatches} (${batch.length} collections)...`);

                await this.processCollectionBatch(batch, force);

                // Delay between batches (except for the last batch)
                if (i + batchSize < collectionsToProcess.length) {
                    console.log(`⏸️  Waiting ${DELAY_BETWEEN_BATCHES / 1000}s before next batch...`);
                    await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES));
                }
            }

            this.logSummary();

        } catch (error) {
            console.error('❌ Batch processing failed:', error.message);
            throw error;
        } finally {
            this.db.close();
        }
    }
}

async function main() {
    const args = process.argv.slice(2);
    
    if (args.includes('--help') || args.includes('-h')) {
        console.log(`
NFT Holders Batch Processor

Usage:
  node batch-process.js [options]

Options:
  --force         Force reprocess all collections (even those already processed)
  --limit N       Limit processing to first N collections (for testing)
  --batch-size N  Number of collections to process per batch (default: ${BATCH_SIZE})
  --help          Show this help

Examples:
  node batch-process.js                    # Process all new collections
  node batch-process.js --limit 5          # Process first 5 collections only
  node batch-process.js --force            # Reprocess all collections
  node batch-process.js --batch-size 3     # Use smaller batches
        `);
        return;
    }

    const options = {};
    
    // Parse command line arguments
    if (args.includes('--force')) {
        options.force = true;
    }
    
    const limitIndex = args.indexOf('--limit');
    if (limitIndex !== -1 && args[limitIndex + 1]) {
        options.limit = parseInt(args[limitIndex + 1]);
    }
    
    const batchSizeIndex = args.indexOf('--batch-size');
    if (batchSizeIndex !== -1 && args[batchSizeIndex + 1]) {
        options.batchSize = parseInt(args[batchSizeIndex + 1]);
    }

    const processor = new BatchProcessor();
    
    try {
        await processor.init();
        await processor.run(options);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { BatchProcessor };