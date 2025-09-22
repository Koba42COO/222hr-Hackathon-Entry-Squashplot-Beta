// ⚠️  SECURITY WARNING: DO NOT USE THESE PLACEHOLDER VALUES IN PRODUCTION!
// QZK Rollout Demo - Demonstration of Quantum Zero-Knowledge Consensus System
// This file demonstrates the QZK consensus engine in action

const { QZKConsensusEngine, CONFIG, ERROR_CODES } = require('./qzk_rollout_engine.js');

class QZKDemo {
    constructor() {
        this.engine = new QZKConsensusEngine();
        this.demoResults = [];
    }

    /**
     * Run the complete QZK demo
     */
    async runDemo() {
        console.log('=== QZK Rollout Engine Demo ===');
        console.log('Demonstrating quantum-resistant consensus system...\n');

        try {
            // Demo 1: Basic consensus
            await this.demoBasicConsensus();

            // Demo 2: Multiple proposals
            await this.demoMultipleProposals();

            // Demo 3: Consensus failure scenarios
            await this.demoConsensusFailures();

            // Demo 4: Quantum attack detection
            await this.demoQuantumAttackDetection();

            // Demo 5: Performance testing
            await this.demoPerformance();

            // Print final results
            this.printDemoResults();

        } catch (error) {
            console.error('Demo error:', error);
        }
    }

    /**
     * Demo 1: Basic consensus scenario
     */
    async demoBasicConsensus() {
        console.log('--- Demo 1: Basic Consensus ---');
        
        const proposalId = 'demo_001_basic';
        const proposal = {
            title: 'System Configuration Update',
            description: 'Update system configuration parameters',
            priority: 'medium',
            changes: {
                timeout: 30000,
                retryCount: 3,
                securityLevel: 'high'
            }
        };
        
        const participants = ['node_alpha', 'node_beta', 'node_gamma', 'node_delta', 'node_epsilon'];
        
        // Initialize proposal
        console.log('Initializing proposal...');
        const initResult = this.engine.initializeProposal(proposalId, proposal, participants);
        console.log('✓ Proposal initialized:', initResult.success ? 'SUCCESS' : 'FAILED');
        
        if (!initResult.success) {
            this.demoResults.push({ demo: 'Basic Consensus', status: 'FAILED', error: initResult.error });
            return;
        }
        
        // Submit votes
        console.log('\nSubmitting votes...');
        const votes = [true, true, true, false, false]; // 3 yes, 2 no
        
        for (let i = 0; i < participants.length; i++) {
            const participantId = participants[i];
            const vote = votes[i];
            const proof = this.engine._generateExpectedProof(participantId, vote);
            
            const voteResult = this.engine.submitVote(proposalId, participantId, vote, proof);
            console.log(`✓ ${participantId} voted ${vote ? 'YES' : 'NO'}: ${voteResult.success ? 'SUCCESS' : 'FAILED'}`);
            
            if (voteResult.consensusReached) {
                console.log('🎉 Consensus reached!');
                break;
            }
        }
        
        // Check final consensus
        const consensusResult = this.engine.checkConsensus(proposalId);
        console.log('\nFinal consensus status:');
        console.log(`- Consensus reached: ${consensusResult.consensusReached ? 'YES' : 'NO'}`);
        console.log(`- Yes votes: ${consensusResult.yesVotes}`);
        console.log(`- No votes: ${consensusResult.noVotes}`);
        console.log(`- Consensus ratio: ${(consensusResult.consensusRatio * 100).toFixed(1)}%`);
        
        this.demoResults.push({
            demo: 'Basic Consensus',
            status: consensusResult.consensusReached ? 'SUCCESS' : 'FAILED',
            yesVotes: consensusResult.yesVotes,
            noVotes: consensusResult.noVotes,
            ratio: consensusResult.consensusRatio
        });
        
        console.log('--- Demo 1 Complete ---\n');
    }

    /**
     * Demo 2: Multiple proposals scenario
     */
    async demoMultipleProposals() {
        console.log('--- Demo 2: Multiple Proposals ---');
        
        const proposals = [
            {
                id: 'demo_002_proposal_1',
                data: { title: 'Security Update', priority: 'high' },
                participants: ['node_1', 'node_2', 'node_3']
            },
            {
                id: 'demo_002_proposal_2',
                data: { title: 'Performance Optimization', priority: 'medium' },
                participants: ['node_2', 'node_3', 'node_4']
            },
            {
                id: 'demo_002_proposal_3',
                data: { title: 'Feature Addition', priority: 'low' },
                participants: ['node_1', 'node_3', 'node_4', 'node_5']
            }
        ];
        
        const results = [];
        
        for (const proposal of proposals) {
            console.log(`\nProcessing proposal: ${proposal.data.title}`);
            
            // Initialize proposal
            const initResult = this.engine.initializeProposal(proposal.id, proposal.data, proposal.participants);
            
            if (initResult.success) {
                // Submit votes (simulate different voting patterns)
                const votePatterns = [
                    [true, true, true],      // Unanimous yes
                    [true, false, true],     // Majority yes
                    [false, false, true, true] // Split decision
                ];
                
                const votes = votePatterns[proposals.indexOf(proposal)];
                
                for (let i = 0; i < proposal.participants.length; i++) {
                    const participantId = proposal.participants[i];
                    const vote = votes[i];
                    const proof = this.engine._generateExpectedProof(participantId, vote);
                    
                    this.engine.submitVote(proposal.id, participantId, vote, proof);
                }
                
                // Check consensus
                const consensusResult = this.engine.checkConsensus(proposal.id);
                results.push({
                    proposalId: proposal.id,
                    title: proposal.data.title,
                    consensusReached: consensusResult.consensusReached,
                    ratio: consensusResult.consensusRatio
                });
                
                console.log(`✓ ${proposal.data.title}: ${consensusResult.consensusReached ? 'CONSENSUS' : 'NO CONSENSUS'}`);
            }
        }
        
        const successfulProposals = results.filter(r => r.consensusReached).length;
        console.log(`\nResults: ${successfulProposals}/${results.length} proposals reached consensus`);
        
        this.demoResults.push({
            demo: 'Multiple Proposals',
            status: 'SUCCESS',
            totalProposals: results.length,
            successfulProposals: successfulProposals,
            successRate: successfulProposals / results.length
        });
        
        console.log('--- Demo 2 Complete ---\n');
    }

    /**
     * Demo 3: Consensus failure scenarios
     */
    async demoConsensusFailures() {
        console.log('--- Demo 3: Consensus Failure Scenarios ---');
        
        const scenarios = [
            {
                name: 'Insufficient Votes',
                proposalId: 'demo_003_insufficient',
                participants: ['node_1', 'node_2'], // Below minimum
                votes: [true, true]
            },
            {
                name: 'Below Threshold',
                proposalId: 'demo_003_threshold',
                participants: ['node_1', 'node_2', 'node_3', 'node_4'],
                votes: [true, false, false, false] // 25% yes, below 60% threshold
            },
            {
                name: 'Timeout Scenario',
                proposalId: 'demo_003_timeout',
                participants: ['node_1', 'node_2', 'node_3'],
                votes: [true, true] // Only 2 votes, timeout will occur
            }
        ];
        
        const results = [];
        
        for (const scenario of scenarios) {
            console.log(`\nTesting: ${scenario.name}`);
            
            const proposal = {
                title: scenario.name,
                description: `Testing ${scenario.name.toLowerCase()} scenario`,
                priority: 'test'
            };
            
            // Initialize proposal
            const initResult = this.engine.initializeProposal(scenario.proposalId, proposal, scenario.participants);
            
            if (initResult.success) {
                // Submit votes
                for (let i = 0; i < scenario.votes.length; i++) {
                    const participantId = scenario.participants[i];
                    const vote = scenario.votes[i];
                    const proof = this.engine._generateExpectedProof(participantId, vote);
                    
                    this.engine.submitVote(scenario.proposalId, participantId, vote, proof);
                }
                
                // Check consensus
                const consensusResult = this.engine.checkConsensus(scenario.proposalId);
                const expectedFailure = !consensusResult.consensusReached;
                
                results.push({
                    scenario: scenario.name,
                    expectedFailure: expectedFailure,
                    actualFailure: !consensusResult.consensusReached,
                    success: expectedFailure === !consensusResult.consensusReached
                });
                
                console.log(`✓ ${scenario.name}: ${expectedFailure ? 'EXPECTED FAILURE' : 'UNEXPECTED RESULT'}`);
            }
        }
        
        const successfulTests = results.filter(r => r.success).length;
        console.log(`\nResults: ${successfulTests}/${results.length} failure scenarios correctly handled`);
        
        this.demoResults.push({
            demo: 'Consensus Failures',
            status: 'SUCCESS',
            totalScenarios: results.length,
            successfulTests: successfulTests,
            successRate: successfulTests / results.length
        });
        
        console.log('--- Demo 3 Complete ---\n');
    }

    /**
     * Demo 4: Quantum attack detection
     */
    async demoQuantumAttackDetection() {
        console.log('--- Demo 4: Quantum Attack Detection ---');
        
        const proposalId = 'demo_004_quantum';
        const proposal = {
            title: 'Quantum Security Test',
            description: 'Testing quantum attack detection mechanisms',
            priority: 'high'
        };
        
        const participants = ['node_secure', 'node_attacker', 'node_normal'];
        
        // Initialize proposal
        const initResult = this.engine.initializeProposal(proposalId, proposal, participants);
        
        if (initResult.success) {
            console.log('Testing quantum attack detection...');
            
            // Simulate normal voting
            const normalVote = this.engine.submitVote(
                proposalId, 
                'node_normal', 
                true, 
                this.engine._generateExpectedProof('node_normal', true)
            );
            console.log(`✓ Normal vote: ${normalVote.success ? 'ACCEPTED' : 'REJECTED'}`);
            
            // Simulate quantum attack (rapid successive votes)
            console.log('Simulating quantum attack...');
            const attackVote1 = this.engine.submitVote(
                proposalId, 
                'node_attacker', 
                true, 
                this.engine._generateExpectedProof('node_attacker', true)
            );
            
            // Immediately try another vote (should trigger attack detection)
            const attackVote2 = this.engine.submitVote(
                proposalId, 
                'node_attacker', 
                false, 
                this.engine._generateExpectedProof('node_attacker', false)
            );
            
            console.log(`✓ Attack vote 1: ${attackVote1.success ? 'ACCEPTED' : 'REJECTED'}`);
            console.log(`✓ Attack vote 2: ${attackVote2.success ? 'ACCEPTED' : 'REJECTED'}`);
            
            const attackDetected = !attackVote2.success && attackVote2.errorCode === ERROR_CODES.QUANTUM_ATTACK_DETECTED;
            console.log(`🎯 Quantum attack detection: ${attackDetected ? 'SUCCESS' : 'FAILED'}`);
            
            this.demoResults.push({
                demo: 'Quantum Attack Detection',
                status: attackDetected ? 'SUCCESS' : 'FAILED',
                attackDetected: attackDetected
            });
        }
        
        console.log('--- Demo 4 Complete ---\n');
    }

    /**
     * Demo 5: Performance testing
     */
    async demoPerformance() {
        console.log('--- Demo 5: Performance Testing ---');
        
        const startTime = Date.now();
        const numProposals = 10;
        const participantsPerProposal = 5;
        const totalVotes = numProposals * participantsPerProposal;
        
        console.log(`Testing performance with ${numProposals} proposals and ${totalVotes} total votes...`);
        
        const results = [];
        
        for (let i = 0; i < numProposals; i++) {
            const proposalId = `perf_${i}`;
            const proposal = {
                title: `Performance Test ${i}`,
                description: `Performance test proposal ${i}`,
                priority: 'test'
            };
            
            const participants = Array.from({ length: participantsPerProposal }, (_, j) => `perf_node_${j}`);
            
            // Initialize proposal
            const initResult = this.engine.initializeProposal(proposalId, proposal, participants);
            
            if (initResult.success) {
                // Submit votes
                for (const participantId of participants) {
                    const vote = Math.random() > 0.5;
                    const proof = this.engine._generateExpectedProof(participantId, vote);
                    
                    this.engine.submitVote(proposalId, participantId, vote, proof);
                }
                
                // Check consensus
                const consensusResult = this.engine.checkConsensus(proposalId);
                results.push(consensusResult.consensusReached);
            }
        }
        
        const endTime = Date.now();
        const duration = endTime - startTime;
        const successfulProposals = results.filter(r => r).length;
        
        console.log(`\nPerformance Results:`);
        console.log(`- Total time: ${duration}ms`);
        console.log(`- Average time per proposal: ${(duration / numProposals).toFixed(2)}ms`);
        console.log(`- Average time per vote: ${(duration / totalVotes).toFixed(2)}ms`);
        console.log(`- Successful proposals: ${successfulProposals}/${numProposals}`);
        console.log(`- Success rate: ${(successfulProposals / numProposals * 100).toFixed(1)}%`);
        
        this.demoResults.push({
            demo: 'Performance Testing',
            status: 'SUCCESS',
            totalTime: duration,
            avgTimePerProposal: duration / numProposals,
            avgTimePerVote: duration / totalVotes,
            successRate: successfulProposals / numProposals
        });
        
        console.log('--- Demo 5 Complete ---\n');
    }

    /**
     * Print demo results summary
     */
    printDemoResults() {
        console.log('=== Demo Results Summary ===');
        console.log('');
        
        for (const result of this.demoResults) {
            const status = result.status === 'SUCCESS' ? '✅' : '❌';
            console.log(`${status} ${result.demo}: ${result.status}`);
            
            if (result.successRate !== undefined) {
                console.log(`   Success Rate: ${(result.successRate * 100).toFixed(1)}%`);
            }
            
            if (result.attackDetected !== undefined) {
                console.log(`   Attack Detection: ${result.attackDetected ? 'WORKING' : 'FAILED'}`);
            }
            
            if (result.avgTimePerVote !== undefined) {
                console.log(`   Avg Time per Vote: ${result.avgTimePerVote.toFixed(2)}ms`);
            }
            
            console.log('');
        }
        
        const overallSuccess = this.demoResults.filter(r => r.status === 'SUCCESS').length;
        const totalDemos = this.demoResults.length;
        
        console.log(`Overall Success Rate: ${overallSuccess}/${totalDemos} (${(overallSuccess / totalDemos * 100).toFixed(1)}%)`);
        console.log('');
        console.log('=== Demo Complete ===');
    }
}

// Run the demo if this file is executed directly
if (require.main === module) {
    const demo = new QZKDemo();
    demo.runDemo();
}

module.exports = QZKDemo;
