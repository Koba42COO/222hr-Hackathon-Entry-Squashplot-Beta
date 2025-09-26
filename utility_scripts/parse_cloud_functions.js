// ⚠️  SECURITY WARNING: DO NOT USE THESE PLACEHOLDER VALUES IN PRODUCTION!
// Dracattus NFT Upgrade System - Parse Cloud Functions
// This file contains cloud functions for handling NFT upgrades in the Dracattus game

const Parse = require('parse/node');

// Configure Parse with your app credentials
Parse.initialize(
    'YOUR_PARSE_APP_ID',           // ⚠️  Replace with your actual Parse App ID
    'YOUR_PARSE_JAVASCRIPT_KEY',   // ⚠️  Replace with your actual Parse JavaScript Key
    'YOUR_PARSE_MASTER_KEY'        // ⚠️  Replace with your actual Parse Master Key
);

Parse.serverURL = 'https://your-parse-server.com/parse'; // ⚠️  Replace with your actual Parse Server URL

// Error codes for consistent error handling
const ERROR_CODES = {
    NFT_NOT_FOUND: 'NFT_001',
    INVALID_SIGNATURE: 'AUTH_001',
    UNAUTHORIZED_WALLET: 'AUTH_002',
    NO_UPGRADE_AVAILABLE: 'UPG_001',
    REQUIREMENTS_NOT_MET: 'UPG_002',
    UPGRADE_IN_PROGRESS: 'UPG_003',
    COOLDOWN_PERIOD: 'UPG_004',
    DATABASE_ERROR: 'SYS_001',
    INVALID_TIMESTAMP: 'AUTH_003',
    INVALID_ACTION: 'AUTH_004'
};

/**
 * Cloud function to check if an NFT can be upgraded
 * @param {Object} request - Parse Cloud request object
 * @param {string} request.params.encoded_id - The NFT's encoded ID
 * @param {boolean} request.params.useMasterKey - For admin access
 * @returns {Object} Upgrade availability and details
 */
Parse.Cloud.define('canUpgrade', async (request) => {
    try {
        const { encoded_id, useMasterKey = false } = request.params;
        
        if (!encoded_id) {
            throw new Error('encoded_id is required');
        }

        // Find the NFT
        const nft = await findNFT(encoded_id, useMasterKey);
        if (!nft) {
            return {
                success: false,
                error: 'NFT not found',
                errorCode: ERROR_CODES.NFT_NOT_FOUND
            };
        }

        // Get upgrade configuration
        const upgradeConfig = await getUpgradeConfig(nft.get('version'));
        if (!upgradeConfig) {
            return {
                success: false,
                error: 'No upgrade available',
                errorCode: ERROR_CODES.NO_UPGRADE_AVAILABLE
            };
        }

        // Check upgrade availability
        const availability = await checkUpgradeAvailability(nft, upgradeConfig);
        
        return {
            success: true,
            data: {
                canUpgrade: availability.canUpgrade,
                currentVersion: nft.get('version'),
                availableVersion: upgradeConfig.version,
                changes: upgradeConfig.changes,
                requirements: upgradeConfig.requirements,
                cooldownRemaining: availability.cooldownRemaining
            }
        };

    } catch (error) {
        console.error('canUpgrade error:', error);
        return {
            success: false,
            error: error.message,
            errorCode: ERROR_CODES.DATABASE_ERROR
        };
    }
});

/**
 * Cloud function to execute NFT upgrade
 * @param {Object} request - Parse Cloud request object
 * @param {Object} request.params - Upgrade parameters
 * @param {string} request.params.encoded_id - NFT encoded ID
 * @param {number} request.params.timestamp - Request timestamp
 * @param {string} request.params.action - Action type (must be 'upgrade')
 * @param {string} request.params.signature - Wallet signature
 * @param {string} request.params.publicKey - Public key
 * @param {string} request.params.wallet - Wallet address
 * @returns {Object} Upgrade result
 */
Parse.Cloud.define('doUpgrade', async (request) => {
    try {
        const { 
            encoded_id, 
            timestamp, 
            action, 
            signature, 
            publicKey, 
            wallet 
        } = request.params;

        // Validate required parameters
        if (!encoded_id || !timestamp || !action || !signature || !publicKey || !wallet) {
            return {
                success: false,
                error: 'Missing required parameters',
                errorCode: ERROR_CODES.INVALID_ACTION
            };
        }

        // Validate action
        if (action !== 'upgrade') {
            return {
                success: false,
                error: 'Invalid action',
                errorCode: ERROR_CODES.INVALID_ACTION
            };
        }

        // Validate timestamp (within 5 minutes)
        const now = Date.now();
        const timeDiff = Math.abs(now - timestamp);
        if (timeDiff > 5 * 60 * 1000) {
            return {
                success: false,
                error: 'Request expired',
                errorCode: ERROR_CODES.INVALID_TIMESTAMP
            };
        }

        // Verify signature (placeholder - implement actual verification)
        const signatureValid = await verifySignature(signature, publicKey, wallet, timestamp, action);
        if (!signatureValid) {
            return {
                success: false,
                error: 'Invalid signature',
                errorCode: ERROR_CODES.INVALID_SIGNATURE
            };
        }

        // Find the NFT
        const nft = await findNFT(encoded_id, false);
        if (!nft) {
            return {
                success: false,
                error: 'NFT not found',
                errorCode: ERROR_CODES.NFT_NOT_FOUND
            };
        }

        // Verify ownership
        if (nft.get('owner_wallet') !== wallet) {
            return {
                success: false,
                error: 'Unauthorized wallet',
                errorCode: ERROR_CODES.UNAUTHORIZED_WALLET
            };
        }

        // Get upgrade configuration
        const upgradeConfig = await getUpgradeConfig(nft.get('version'));
        if (!upgradeConfig) {
            return {
                success: false,
                error: 'No upgrade available',
                errorCode: ERROR_CODES.NO_UPGRADE_AVAILABLE
            };
        }

        // Check upgrade availability
        const availability = await checkUpgradeAvailability(nft, upgradeConfig);
        if (!availability.canUpgrade) {
            return {
                success: false,
                error: availability.reason,
                errorCode: availability.errorCode
            };
        }

        // Execute upgrade in transaction
        const result = await Parse.Cloud.run('executeUpgradeTransaction', {
            nftId: nft.id,
            upgradeConfig: upgradeConfig,
            wallet: wallet,
            timestamp: now
        });

        return {
            success: true,
            data: {
                nft: {
                    encoded_id: nft.get('encoded_id'),
                    name: nft.get('name'),
                    version: upgradeConfig.version,
                    attributes: result.attributes
                },
                upgradedFrom: nft.get('version'),
                upgradedTo: upgradeConfig.version,
                timestamp: now,
                transactionHash: result.transactionHash
            }
        };

    } catch (error) {
        console.error('doUpgrade error:', error);
        return {
            success: false,
            error: error.message,
            errorCode: ERROR_CODES.DATABASE_ERROR
        };
    }
});

// Helper functions
async function findNFT(encodedId, useMasterKey) {
    const query = new Parse.Query('DRACATTUS');
    query.equalTo('encoded_id', encodedId);
    
    if (useMasterKey) {
        query.useMasterKey();
    }
    
    return await query.first();
}

async function getUpgradeConfig(currentVersion) {
    const query = new Parse.Query('UPGRADE_CONFIG');
    query.equalTo('from_version', currentVersion);
    query.ascending('version');
    
    return await query.first();
}

async function checkUpgradeAvailability(nft, upgradeConfig) {
    // Check cooldown period
    const lastUpgrade = nft.get('last_upgrade');
    if (lastUpgrade) {
        const cooldownPeriod = upgradeConfig.get('cooldown_period') || 24 * 60 * 60 * 1000; // 24 hours default
        const timeSinceUpgrade = Date.now() - lastUpgrade.getTime();
        
        if (timeSinceUpgrade < cooldownPeriod) {
            return {
                canUpgrade: false,
                reason: 'Cooldown period active',
                errorCode: ERROR_CODES.COOLDOWN_PERIOD,
                cooldownRemaining: cooldownPeriod - timeSinceUpgrade
            };
        }
    }

    // Check requirements (placeholder - implement actual requirement checking)
    const requirements = upgradeConfig.get('requirements');
    if (requirements) {
        // Implement requirement validation logic here
        // For now, assume requirements are met
    }

    return {
        canUpgrade: true,
        cooldownRemaining: 0
    };
}

async function verifySignature(signature, publicKey, wallet, timestamp, action) {
    // Placeholder - implement actual signature verification
    // This should verify the signature against the provided parameters
    return true;
}

// Transaction function for atomic upgrade
Parse.Cloud.define('executeUpgradeTransaction', async (request) => {
    const { nftId, upgradeConfig, wallet, timestamp } = request.params;
    
    try {
        // Get the NFT
        const nft = await new Parse.Query('DRACATTUS').get(nftId);
        
        // Apply upgrade changes
        const changes = upgradeConfig.get('changes');
        const currentAttributes = nft.get('attributes') || {};
        
        // Merge attribute changes
        const newAttributes = { ...currentAttributes, ...changes.attributes };
        
        // Update NFT
        nft.set('version', upgradeConfig.get('version'));
        nft.set('attributes', newAttributes);
        nft.set('last_upgrade', new Date(timestamp));
        
        // Add to upgrade history
        const upgradeHistory = nft.get('upgrade_history') || [];
        upgradeHistory.push({
            from_version: nft.get('version'),
            to_version: upgradeConfig.get('version'),
            timestamp: timestamp,
            wallet: wallet,
            changes: changes
        });
        nft.set('upgrade_history', upgradeHistory);
        
        // Save NFT
        await nft.save(null, { useMasterKey: true });
        
        // Log upgrade in UPGRADE_HISTORY collection
        const historyEntry = new Parse.Object('UPGRADE_HISTORY');
        historyEntry.set('encoded_id', nft.get('encoded_id'));
        historyEntry.set('from_version', nft.get('version'));
        historyEntry.set('to_version', upgradeConfig.get('version'));
        historyEntry.set('timestamp', new Date(timestamp));
        historyEntry.set('wallet', wallet);
        historyEntry.set('changes_applied', changes);
        await historyEntry.save(null, { useMasterKey: true });
        
        return {
            attributes: newAttributes,
            transactionHash: `upgrade_${nftId}_${timestamp}`
        };
        
    } catch (error) {
        throw new Error(`Upgrade transaction failed: ${error.message}`);
    }
});

module.exports = {
    ERROR_CODES,
    canUpgrade: Parse.Cloud.define('canUpgrade'),
    doUpgrade: Parse.Cloud.define('doUpgrade')
};
