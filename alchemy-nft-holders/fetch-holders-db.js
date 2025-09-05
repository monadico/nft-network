const { Alchemy, Network } = require("alchemy-sdk");
const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
require('dotenv').config();

// Database setup
const DB_PATH = 'data/holders.sqlite';
const DATA_DIR = 'data';

// Alchemy configuration - Monad testnet
const config = {
    apiKey: process.env.ALCHEMY_API_KEY || "<YOUR_ALCHEMY_API_KEY>",
    network: Network.MONAD_TESTNET,
};
const alchemy = new Alchemy(config);

// Database schema (matching the Python backend structure)
const SCHEMA_SQL = `
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT,
    token_count INTEGER,
    owner_count INTEGER,
    total_quantity INTEGER,
    token_standard TEXT,
    processed_at TEXT
);

CREATE TABLE IF NOT EXISTS collection_holders (
    collection_id TEXT,
    holder_address TEXT,
    token_count INTEGER DEFAULT 1,
    PRIMARY KEY (collection_id, holder_address)
);

CREATE INDEX IF NOT EXISTS idx_collection_holders_collection ON collection_holders(collection_id);
CREATE INDEX IF NOT EXISTS idx_collection_holders_address ON collection_holders(holder_address);
`;

class HoldersDatabase {
    constructor() {
        this.db = null;
    }

    async init() {
        // Ensure data directory exists
        if (!fs.existsSync(DATA_DIR)) {
            fs.mkdirSync(DATA_DIR, { recursive: true });
        }

        return new Promise((resolve, reject) => {
            this.db = new sqlite3.Database(DB_PATH, (err) => {
                if (err) {
                    reject(err);
                } else {
                    console.log('📚 Connected to SQLite database');
                    this.createSchema().then(resolve).catch(reject);
                }
            });
        });
    }

    async createSchema() {
        return new Promise((resolve, reject) => {
            this.db.exec(SCHEMA_SQL, (err) => {
                if (err) {
                    reject(err);
                } else {
                    console.log('✅ Database schema initialized');
                    resolve();
                }
            });
        });
    }

    async saveCollectionHolders(collectionId, name, holders, tokenStandard = 'ERC-721') {
        const processedAt = new Date().toISOString();
        const ownerCount = holders.length;
        
        return new Promise((resolve, reject) => {
            this.db.serialize(() => {
                this.db.run('BEGIN TRANSACTION');

                // Clear existing holders for this collection
                this.db.run(
                    'DELETE FROM collection_holders WHERE collection_id = ?',
                    [collectionId],
                    (err) => {
                        if (err) {
                            this.db.run('ROLLBACK');
                            return reject(err);
                        }
                    }
                );

                // Insert new holders (assuming 1 token each)
                const stmt = this.db.prepare(`
                    INSERT INTO collection_holders (collection_id, holder_address, token_count) 
                    VALUES (?, ?, ?)
                `);
                
                holders.forEach((holder) => {
                    stmt.run([collectionId, holder, 1]); // Each holder owns 1+ NFTs
                });
                
                stmt.finalize((err) => {
                    if (err) {
                        this.db.run('ROLLBACK');
                        return reject(err);
                    }
                });

                // Update collection info
                this.db.run(`
                    INSERT INTO collections (collection_id, name, token_count, owner_count, total_quantity, token_standard, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection_id) DO UPDATE SET
                        name = excluded.name,
                        owner_count = excluded.owner_count,
                        token_standard = excluded.token_standard,
                        processed_at = excluded.processed_at
                `, [collectionId, name, null, ownerCount, ownerCount, tokenStandard, processedAt], (err) => {
                    if (err) {
                        this.db.run('ROLLBACK');
                        return reject(err);
                    } else {
                        this.db.run('COMMIT', (err) => {
                            if (err) {
                                return reject(err);
                            } else {
                                console.log(`💾 Saved ${ownerCount} holders for ${name} (${collectionId})`);
                                resolve({ collectionId, ownerCount });
                            }
                        });
                    }
                });
            });
        });
    }

    async getCollectionInfo(collectionId) {
        return new Promise((resolve, reject) => {
            this.db.get(`
                SELECT collection_id, name, token_count, owner_count, processed_at
                FROM collections 
                WHERE collection_id = ?
            `, [collectionId], (err, row) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(row);
                }
            });
        });
    }

    async getAllCollections() {
        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT collection_id, name, owner_count, processed_at
                FROM collections
                ORDER BY owner_count DESC
            `, (err, rows) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(rows);
                }
            });
        });
    }

    close() {
        if (this.db) {
            this.db.close();
            console.log('📚 Database connection closed');
        }
    }
}

async function fetchHoldersForCollection(collectionAddress, collectionName = null) {
    try {
        console.log(`🔍 Fetching holders for collection: ${collectionAddress}`);
        console.log('This may take a moment...\n');
        
        const owners = await alchemy.nft.getOwnersForContract(collectionAddress);
        
        console.log(`Found ${owners.owners.length} unique holders`);
        console.log('First 10 holders:');
        owners.owners.slice(0, 10).forEach((owner, index) => {
            console.log(`${index + 1}. ${owner}`);
        });
        
        if (owners.owners.length > 10) {
            console.log(`... and ${owners.owners.length - 10} more holders`);
        }
        
        return {
            holders: owners.owners,
            totalCount: owners.owners.length
        };
        
    } catch (error) {
        console.error(`❌ Error fetching holders for ${collectionAddress}:`, error.message);
        throw error;
    }
}

async function processCollection(db, collectionId, collectionName = null, force = false) {
    try {
        // Check if already processed (unless force is true)
        if (!force) {
            const existing = await db.getCollectionInfo(collectionId);
            if (existing) {
                console.log(`⚠️  Collection ${collectionName || collectionId} already processed on ${existing.processed_at}`);
                console.log(`   Found ${existing.owner_count} holders. Use --force to reprocess.`);
                return existing;
            }
        }

        const result = await fetchHoldersForCollection(collectionId, collectionName);
        const saved = await db.saveCollectionHolders(
            collectionId, 
            collectionName || `Collection ${collectionId.slice(0, 10)}...`, 
            result.holders
        );
        
        return saved;
        
    } catch (error) {
        console.error(`❌ Failed to process collection ${collectionId}:`, error.message);
        throw error;
    }
}

// Main execution functions
async function fetchSingleCollection(collectionAddress, collectionName) {
    const db = new HoldersDatabase();
    
    try {
        await db.init();
        const result = await processCollection(db, collectionAddress, collectionName);
        console.log(`\n✅ Successfully processed collection: ${result.ownerCount} holders saved`);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    } finally {
        db.close();
    }
}

async function listCollections() {
    const db = new HoldersDatabase();
    
    try {
        await db.init();
        const collections = await db.getAllCollections();
        
        if (collections.length === 0) {
            console.log('📭 No collections found in database');
            return;
        }
        
        console.log(`\n📊 Found ${collections.length} collections in database:\n`);
        collections.forEach((col, index) => {
            console.log(`${index + 1}. ${col.name}`);
            console.log(`   ID: ${col.collection_id}`);
            console.log(`   Holders: ${col.owner_count}`);
            console.log(`   Processed: ${col.processed_at}`);
            console.log('');
        });
        
    } catch (error) {
        console.error('❌ Error:', error.message);
    } finally {
        db.close();
    }
}

// Command line interface
async function main() {
    const args = process.argv.slice(2);
    
    if (args.includes('--help') || args.includes('-h')) {
        console.log(`
NFT Holders Database Tool (Alchemy API)

Usage:
  node fetch-holders-db.js <collection-address> [collection-name]  # Fetch holders for a collection
  node fetch-holders-db.js --list                                  # List all processed collections
  node fetch-holders-db.js --help                                  # Show this help

Examples:
  node fetch-holders-db.js 0x26c86f2835c114571df2b6ce9ba52296cc0fa6bb "Lil Chogstars"
  node fetch-holders-db.js --list

Note: Requires ALCHEMY_API_KEY environment variable
        `);
        return;
    }
    
    if (args.includes('--list')) {
        await listCollections();
        return;
    }
    
    if (args.length === 0) {
        console.log('❌ Please provide a collection address. Use --help for usage info.');
        process.exit(1);
    }
    
    const collectionAddress = args[0];
    const collectionName = args[1];
    
    // Validate Ethereum address format
    if (!/^0x[a-fA-F0-9]{40}$/.test(collectionAddress)) {
        console.log('❌ Invalid Ethereum address format');
        process.exit(1);
    }
    
    await fetchSingleCollection(collectionAddress, collectionName);
}

// Export for potential use as module
module.exports = {
    HoldersDatabase,
    fetchHoldersForCollection,
    processCollection
};

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}