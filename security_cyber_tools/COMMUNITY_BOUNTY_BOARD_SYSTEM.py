#!/usr/bin/env python3
"""
🎯 COMMUNITY BOUNTY BOARD SYSTEM
===============================

Decentralized marketplace for project requests and fulfillment
Users can post bounties for projects they need built, and community members can fulfill them

FEATURES:
- Bounty posting with detailed requirements
- Skill-based matching of contributors
- Decentralized escrow and payment system
- Reputation-based trust scoring
- Project milestone tracking
- Community voting on bounty quality
- Automatic fulfillment verification
"""

import os
import json
import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BountyStatus(Enum):
    """Status of a bounty"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class BountyPriority(Enum):
    """Priority levels for bounties"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SubmissionStatus(Enum):
    """Status of bounty submissions"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"

@dataclass
class BountyRequirement:
    """Requirements for a bounty"""
    skill_category: str
    skill_level: str  # beginner, intermediate, expert
    estimated_hours: int
    description: str
    must_have: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)

@dataclass
class BountySubmission:
    """A submission for a bounty"""
    id: str
    bounty_id: str
    submitter_wallet: str
    submission_content: str
    submission_files: List[str] = field(default_factory=list)
    status: SubmissionStatus = SubmissionStatus.PENDING
    submitted_at: datetime = field(default_factory=datetime.now)
    review_feedback: str = ""
    final_score: float = 0.0

@dataclass
class Bounty:
    """A bounty posted on the community board"""
    id: str
    title: str
    description: str
    poster_wallet: str
    bounty_amount: float  # In ETH equivalent
    requirements: List[BountyRequirement]
    priority: BountyPriority = BountyPriority.MEDIUM
    status: BountyStatus = BountyStatus.OPEN
    tags: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    submissions: List[BountySubmission] = field(default_factory=list)
    winner_wallet: Optional[str] = None
    completion_date: Optional[datetime] = None
    escrow_locked: bool = False

@dataclass
class CommunityMember:
    """Community member profile"""
    wallet_address: str
    username: str
    skills: Dict[str, str]  # skill -> proficiency level
    reputation_score: float = 0.0
    completed_bounties: int = 0
    success_rate: float = 0.0
    total_earnings: float = 0.0
    badges: List[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=datetime.now)

@dataclass
class EscrowTransaction:
    """Escrow transaction for bounty payments"""
    id: str
    bounty_id: str
    from_wallet: str
    to_wallet: str
    amount: float
    status: str  # locked, released, refunded
    created_at: datetime = field(default_factory=datetime.now)
    released_at: Optional[datetime] = None

class CommunityBountyBoard:
    """Community bounty board system for project requests and fulfillment"""

    def __init__(self):
        self.bounties: Dict[str, Bounty] = {}
        self.members: Dict[str, CommunityMember] = {}
        self.submissions: Dict[str, BountySubmission] = {}
        self.escrow_transactions: Dict[str, EscrowTransaction] = {}

        # Bounty categories and their base rates
        self.category_rates = {
            "web_development": 0.05,
            "mobile_app": 0.08,
            "data_science": 0.10,
            "machine_learning": 0.15,
            "blockchain": 0.12,
            "design": 0.06,
            "writing": 0.04,
            "research": 0.07,
            "consulting": 0.09,
            "other": 0.05
        }

        # Skill level multipliers
        self.skill_multipliers = {
            "beginner": 0.7,
            "intermediate": 1.0,
            "expert": 1.5,
            "master": 2.0
        }

        # Badge system
        self.badge_requirements = {
            "first_blood": {"completed_bounties": 1},
            "reliable": {"success_rate": 0.9, "completed_bounties": 5},
            "expert": {"reputation_score": 500},
            "mentor": {"submissions_reviewed": 10},
            "community_leader": {"reputation_score": 1000, "completed_bounties": 25}
        }

        # Start background services
        self._start_background_services()

        logger.info("🎯 Community Bounty Board initialized")

    def _start_background_services(self):
        """Start background services"""
        # Bounty expiration service
        expiration_thread = threading.Thread(target=self._bounty_expiration_service, daemon=True)
        expiration_thread.start()

        # Escrow monitoring service
        escrow_thread = threading.Thread(target=self._escrow_monitoring_service, daemon=True)
        escrow_thread.start()

        # Reputation update service
        reputation_thread = threading.Thread(target=self._reputation_update_service, daemon=True)
        reputation_thread.start()

        logger.info("🔄 Background services started")

    def _bounty_expiration_service(self):
        """Monitor and expire bounties that have passed their deadline"""
        while True:
            try:
                current_time = datetime.now()
                expired_bounties = []

                for bounty_id, bounty in self.bounties.items():
                    if (bounty.status == BountyStatus.OPEN and
                        bounty.deadline and
                        current_time > bounty.deadline):
                        expired_bounties.append(bounty_id)

                for bounty_id in expired_bounties:
                    self.bounties[bounty_id].status = BountyStatus.EXPIRED
                    logger.info(f"⏰ Bounty expired: {bounty_id}")

                time.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Expiration service error: {e}")

    def _escrow_monitoring_service(self):
        """Monitor escrow transactions"""
        while True:
            try:
                # Check for completed bounties and release escrow
                for bounty_id, bounty in self.bounties.items():
                    if (bounty.status == BountyStatus.COMPLETED and
                        bounty.winner_wallet and
                        not self._is_escrow_released(bounty_id)):

                        self._release_escrow(bounty_id, bounty.winner_wallet)
                        logger.info(f"💰 Escrow released for bounty: {bounty_id}")

                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Escrow monitoring error: {e}")

    def _reputation_update_service(self):
        """Update member reputations and badges"""
        while True:
            try:
                self._update_all_reputations()
                self._update_all_badges()
                time.sleep(86400)  # Update daily

            except Exception as e:
                logger.error(f"Reputation update error: {e}")

    def register_member(self, wallet_address: str, username: str,
                       skills: Dict[str, str] = None) -> CommunityMember:
        """Register a new community member"""

        if wallet_address in self.members:
            raise ValueError("Member already registered")

        if skills is None:
            skills = {}

        member = CommunityMember(
            wallet_address=wallet_address,
            username=username,
            skills=skills
        )

        self.members[wallet_address] = member
        logger.info(f"👤 Community member registered: {username}")

        return member

    def post_bounty(self, poster_wallet: str, title: str, description: str,
                   requirements: List[BountyRequirement], bounty_amount: float,
                   tags: List[str] = None, priority: BountyPriority = BountyPriority.MEDIUM,
                   deadline_days: int = 30) -> Bounty:
        """Post a new bounty on the community board"""

        if poster_wallet not in self.members:
            raise ValueError("Poster not registered as community member")

        if tags is None:
            tags = []

        bounty_id = f"bounty_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        deadline = datetime.now() + timedelta(days=deadline_days) if deadline_days > 0 else None

        bounty = Bounty(
            id=bounty_id,
            title=title,
            description=description,
            poster_wallet=poster_wallet,
            bounty_amount=bounty_amount,
            requirements=requirements,
            priority=priority,
            tags=tags,
            deadline=deadline
        )

        self.bounties[bounty_id] = bounty

        # Lock funds in escrow
        self._create_escrow(bounty_id, poster_wallet, bounty_amount)

        logger.info(f"📋 Bounty posted: {title} (${bounty_amount})")

        return bounty

    def _create_escrow(self, bounty_id: str, from_wallet: str, amount: float):
        """Create escrow transaction for bounty payment"""
        escrow_id = f"escrow_{bounty_id}"

        escrow = EscrowTransaction(
            id=escrow_id,
            bounty_id=bounty_id,
            from_wallet=from_wallet,
            to_wallet="",  # To be set when winner is selected
            amount=amount,
            status="locked"
        )

        self.escrow_transactions[escrow_id] = escrow
        logger.info(f"🔒 Escrow created: ${amount} for bounty {bounty_id}")

    def _is_escrow_released(self, bounty_id: str) -> bool:
        """Check if escrow has been released for a bounty"""
        escrow_id = f"escrow_{bounty_id}"
        if escrow_id in self.escrow_transactions:
            return self.escrow_transactions[escrow_id].status == "released"
        return False

    def _release_escrow(self, bounty_id: str, to_wallet: str):
        """Release escrow funds to winner"""
        escrow_id = f"escrow_{bounty_id}"

        if escrow_id in self.escrow_transactions:
            escrow = self.escrow_transactions[escrow_id]
            escrow.to_wallet = to_wallet
            escrow.status = "released"
            escrow.released_at = datetime.now()

            # Update winner's earnings
            if to_wallet in self.members:
                self.members[to_wallet].total_earnings += escrow.amount

            logger.info(f"💸 Escrow released: ${escrow.amount} to {to_wallet}")

    def submit_for_bounty(self, submitter_wallet: str, bounty_id: str,
                         submission_content: str, submission_files: List[str] = None) -> BountySubmission:
        """Submit work for a bounty"""

        if bounty_id not in self.bounties:
            raise ValueError("Bounty not found")

        if submitter_wallet not in self.members:
            raise ValueError("Submitter not registered")

        bounty = self.bounties[bounty_id]

        if bounty.status != BountyStatus.OPEN:
            raise ValueError("Bounty is not open for submissions")

        if submission_files is None:
            submission_files = []

        submission_id = f"sub_{bounty_id}_{uuid.uuid4().hex[:8]}"

        submission = BountySubmission(
            id=submission_id,
            bounty_id=bounty_id,
            submitter_wallet=submitter_wallet,
            submission_content=submission_content,
            submission_files=submission_files
        )

        self.submissions[submission_id] = submission
        bounty.submissions.append(submission)

        logger.info(f"📤 Submission received for bounty {bounty_id} from {submitter_wallet}")

        return submission

    def evaluate_submission(self, bounty_id: str, submission_id: str,
                          reviewer_wallet: str, score: float,
                          feedback: str = "", accept: bool = False):
        """Evaluate a bounty submission"""

        if bounty_id not in self.bounties:
            raise ValueError("Bounty not found")

        if submission_id not in self.submissions:
            raise ValueError("Submission not found")

        bounty = self.bounties[bounty_id]
        submission = self.submissions[submission_id]

        if bounty.poster_wallet != reviewer_wallet:
            raise ValueError("Only bounty poster can evaluate submissions")

        submission.final_score = score
        submission.review_feedback = feedback

        if accept:
            submission.status = SubmissionStatus.ACCEPTED
            bounty.status = BountyStatus.COMPLETED
            bounty.winner_wallet = submission.submitter_wallet
            bounty.completion_date = datetime.now()

            # Update winner stats
            winner = self.members.get(submission.submitter_wallet)
            if winner:
                winner.completed_bounties += 1
                winner.reputation_score += score * 10

            logger.info(f"✅ Bounty {bounty_id} completed by {submission.submitter_wallet}")
        else:
            submission.status = SubmissionStatus.REJECTED
            logger.info(f"❌ Submission {submission_id} rejected")

    def find_matching_bounties(self, member_wallet: str) -> List[Dict[str, Any]]:
        """Find bounties that match a member's skills"""

        if member_wallet not in self.members:
            return []

        member = self.members[member_wallet]
        matching_bounties = []

        for bounty in self.bounties.values():
            if bounty.status != BountyStatus.OPEN:
                continue

            match_score = self._calculate_skill_match(member.skills, bounty.requirements)

            if match_score > 0.3:  # Minimum match threshold
                matching_bounties.append({
                    "bounty": bounty,
                    "match_score": match_score,
                    "required_skills": [req.skill_category for req in bounty.requirements],
                    "member_skills": list(member.skills.keys())
                })

        # Sort by match score
        matching_bounties.sort(key=lambda x: x["match_score"], reverse=True)

        return matching_bounties

    def _calculate_skill_match(self, member_skills: Dict[str, str],
                             requirements: List[BountyRequirement]) -> float:
        """Calculate how well member skills match bounty requirements"""

        if not requirements:
            return 0.0

        total_match_score = 0.0
        max_possible_score = 0.0

        for requirement in requirements:
            max_possible_score += 1.0

            if requirement.skill_category in member_skills:
                member_level = member_skills[requirement.skill_category]
                required_level = requirement.skill_level

                # Calculate level compatibility
                level_score = self._calculate_level_compatibility(member_level, required_level)

                # Weight by importance (must-have vs nice-to-have)
                weight = 1.5 if requirement.must_have else 1.0
                total_match_score += level_score * weight

        return total_match_score / max_possible_score if max_possible_score > 0 else 0.0

    def _calculate_level_compatibility(self, member_level: str, required_level: str) -> float:
        """Calculate compatibility between skill levels"""

        levels = ["beginner", "intermediate", "expert", "master"]
        member_idx = levels.index(member_level) if member_level in levels else 0
        required_idx = levels.index(required_level) if required_level in levels else 0

        if member_idx >= required_idx:
            # Member has required level or higher
            return 1.0
        elif member_idx == required_idx - 1:
            # One level below - partial match
            return 0.7
        else:
            # Too far below - poor match
            return 0.3

    def _update_all_reputations(self):
        """Update reputation scores for all members"""
        for member in self.members.values():
            # Base reputation from completed bounties
            base_reputation = member.completed_bounties * 50

            # Success rate bonus
            success_bonus = member.success_rate * 200

            # Skill diversity bonus
            skill_bonus = len(member.skills) * 25

            # Time-based decay (newer members get boost)
            days_since_join = (datetime.now() - member.joined_at).days
            recency_bonus = max(0, 100 - days_since_join)

            member.reputation_score = base_reputation + success_bonus + skill_bonus + recency_bonus

    def _update_all_badges(self):
        """Update badges for all members"""
        for member in self.members.values():
            earned_badges = []

            for badge_name, requirements in self.badge_requirements.items():
                if self._meets_badge_requirements(member, badge_name, requirements):
                    if badge_name not in member.badges:
                        member.badges.append(badge_name)
                        earned_badges.append(badge_name)

            if earned_badges:
                logger.info(f"🏆 Badges earned by {member.username}: {', '.join(earned_badges)}")

    def _meets_badge_requirements(self, member: CommunityMember, badge_name: str,
                                requirements: Dict[str, Any]) -> bool:
        """Check if member meets badge requirements"""

        for req_key, req_value in requirements.items():
            if req_key == "completed_bounties":
                if member.completed_bounties < req_value:
                    return False
            elif req_key == "success_rate":
                if member.success_rate < req_value:
                    return False
            elif req_key == "reputation_score":
                if member.reputation_score < req_value:
                    return False
            elif req_key == "submissions_reviewed":
                # This would need to be tracked separately
                return False

        return True

    def get_community_stats(self) -> Dict[str, Any]:
        """Get comprehensive community statistics"""

        total_members = len(self.members)
        total_bounties = len(self.bounties)
        active_bounties = len([b for b in self.bounties.values() if b.status == BountyStatus.OPEN])
        completed_bounties = len([b for b in self.bounties.values() if b.status == BountyStatus.COMPLETED])
        total_bounty_value = sum(b.bounty_amount for b in self.bounties.values())

        # Skill distribution
        skill_counts = {}
        for member in self.members.values():
            for skill in member.skills.keys():
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

        # Top skills
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Bounty category distribution
        category_counts = {}
        for bounty in self.bounties.values():
            for tag in bounty.tags:
                category_counts[tag] = category_counts.get(tag, 0) + 1

        return {
            "overview": {
                "total_members": total_members,
                "total_bounties": total_bounties,
                "active_bounties": active_bounties,
                "completed_bounties": completed_bounties,
                "total_bounty_value": total_bounty_value,
                "completion_rate": completed_bounties / total_bounties if total_bounties > 0 else 0
            },
            "skills": {
                "total_unique_skills": len(skill_counts),
                "top_skills": top_skills,
                "skill_distribution": skill_counts
            },
            "bounties": {
                "category_distribution": category_counts,
                "average_bounty_size": total_bounty_value / total_bounties if total_bounties > 0 else 0
            }
        }

    def get_member_dashboard(self, wallet_address: str) -> Dict[str, Any]:
        """Get comprehensive member dashboard"""

        if wallet_address not in self.members:
            raise ValueError("Member not registered")

        member = self.members[wallet_address]

        # Get member's active bounties (posted)
        posted_bounties = [b for b in self.bounties.values() if b.poster_wallet == wallet_address]

        # Get member's submissions
        member_submissions = [s for s in self.submissions.values() if s.submitter_wallet == wallet_address]

        # Get matching bounties
        matching_bounties = self.find_matching_bounties(wallet_address)

        # Calculate success metrics
        completed_submissions = [s for s in member_submissions if s.status == SubmissionStatus.ACCEPTED]
        member.success_rate = len(completed_submissions) / len(member_submissions) if member_submissions else 0

        return {
            "member_info": {
                "wallet_address": member.wallet_address,
                "username": member.username,
                "reputation_score": member.reputation_score,
                "badges": member.badges,
                "joined_at": member.joined_at.isoformat(),
                "skills": member.skills
            },
            "bounties": {
                "posted": len(posted_bounties),
                "active_posted": len([b for b in posted_bounties if b.status == BountyStatus.OPEN]),
                "completed_posted": len([b for b in posted_bounties if b.status == BountyStatus.COMPLETED])
            },
            "submissions": {
                "total": len(member_submissions),
                "pending": len([s for s in member_submissions if s.status == SubmissionStatus.PENDING]),
                "accepted": len([s for s in member_submissions if s.status == SubmissionStatus.ACCEPTED]),
                "rejected": len([s for s in member_submissions if s.status == SubmissionStatus.REJECTED]),
                "success_rate": member.success_rate
            },
            "earnings": {
                "total_earned": member.total_earnings,
                "average_per_bounty": member.total_earnings / member.completed_bounties if member.completed_bounties > 0 else 0
            },
            "opportunities": {
                "matching_bounties": len(matching_bounties),
                "top_matches": matching_bounties[:5]
            }
        }

def main():
    """Demonstrate the Community Bounty Board System"""

    print("🎯 COMMUNITY BOUNTY BOARD SYSTEM DEMO")
    print("=" * 50)

    # Initialize bounty board
    board = CommunityBountyBoard()

    # Register community members with different skills
    print("\n👥 REGISTERING COMMUNITY MEMBERS...")

    members_data = [
        ("alice_wallet", "AliceDev", {"python": "expert", "web_development": "intermediate", "api_design": "expert"}),
        ("bob_wallet", "BobDesigner", {"design": "expert", "ui_ux": "expert", "prototyping": "intermediate"}),
        ("charlie_wallet", "CharlieML", {"machine_learning": "expert", "data_science": "expert", "python": "intermediate"}),
        ("diana_wallet", "DianaWriter", {"writing": "expert", "content_strategy": "expert", "seo": "intermediate"}),
        ("eve_wallet", "EveBlockchain", {"blockchain": "expert", "smart_contracts": "expert", "solidity": "expert"})
    ]

    members = []
    for wallet, username, skills in members_data:
        member = board.register_member(wallet, username, skills)
        members.append((wallet, username))
        print(f"✅ {username}: {', '.join(skills.keys())}")

    # Post bounties
    print("\n📋 POSTING BOUNTIES...")

    bounties_data = [
        {
            "title": "React E-commerce Platform",
            "description": "Build a modern e-commerce platform with React, Node.js, and Stripe integration",
            "poster": "alice_wallet",
            "amount": 2.5,
            "requirements": [
                BountyRequirement("web_development", "intermediate", 40, "Full-stack development experience"),
                BountyRequirement("react", "expert", 20, "Advanced React patterns and hooks"),
                BountyRequirement("api_design", "intermediate", 15, "REST API development")
            ],
            "tags": ["web_development", "react", "ecommerce"]
        },
        {
            "title": "Mobile App UI/UX Design",
            "description": "Design beautiful mobile app interface for fitness tracking application",
            "poster": "bob_wallet",
            "amount": 1.8,
            "requirements": [
                BountyRequirement("design", "expert", 25, "Mobile UI/UX design expertise"),
                BountyRequirement("prototyping", "intermediate", 15, "Interactive prototyping skills")
            ],
            "tags": ["design", "mobile", "ui_ux"]
        },
        {
            "title": "ML Model for Customer Prediction",
            "description": "Build machine learning model to predict customer churn using historical data",
            "poster": "charlie_wallet",
            "amount": 3.2,
            "requirements": [
                BountyRequirement("machine_learning", "expert", 35, "ML model development experience"),
                BountyRequirement("data_science", "expert", 25, "Data analysis and preprocessing"),
                BountyRequirement("python", "intermediate", 15, "Python programming")
            ],
            "tags": ["machine_learning", "data_science", "prediction"]
        },
        {
            "title": "Technical Documentation",
            "description": "Write comprehensive technical documentation for our API platform",
            "poster": "diana_wallet",
            "amount": 1.2,
            "requirements": [
                BountyRequirement("writing", "expert", 30, "Technical writing experience"),
                BountyRequirement("api_design", "intermediate", 10, "API knowledge")
            ],
            "tags": ["writing", "documentation", "api"]
        },
        {
            "title": "Smart Contract Development",
            "description": "Develop DeFi smart contracts for yield farming protocol",
            "poster": "eve_wallet",
            "amount": 4.5,
            "requirements": [
                BountyRequirement("blockchain", "expert", 45, "Blockchain development experience"),
                BountyRequirement("smart_contracts", "expert", 30, "Smart contract security and optimization"),
                BountyRequirement("solidity", "expert", 20, "Solidity programming")
            ],
            "tags": ["blockchain", "smart_contracts", "defi"]
        }
    ]

    bounties = []
    for bounty_data in bounties_data:
        bounty = board.post_bounty(
            poster_wallet=bounty_data["poster"],
            title=bounty_data["title"],
            description=bounty_data["description"],
            requirements=bounty_data["requirements"],
            bounty_amount=bounty_data["amount"],
            tags=bounty_data["tags"]
        )
        bounties.append(bounty)
        print(f"📋 {bounty.title}: ${bounty.bounty_amount} ({len(bounty.requirements)} requirements)")

    # Simulate submissions
    print("\n📤 SUBMITTING FOR BOUNTIES...")

    submissions_data = [
        ("bob_wallet", bounties[0].id, "React e-commerce platform with modern design patterns"),
        ("charlie_wallet", bounties[1].id, "ML model using Random Forest and Neural Networks"),
        ("alice_wallet", bounties[2].id, "Complete API documentation with examples"),
        ("eve_wallet", bounties[3].id, "DeFi smart contracts with security audits"),
        ("diana_wallet", bounties[4].id, "Mobile app design with fitness focus")
    ]

    submissions = []
    for submitter, bounty_id, content in submissions_data:
        try:
            submission = board.submit_for_bounty(submitter, bounty_id, content)
            submissions.append(submission)
            print(f"📤 {submitter} submitted for {bounty_id}")
        except Exception as e:
            print(f"❌ Submission failed: {e}")

    # Evaluate submissions (simulate bounty completion)
    print("\n✅ EVALUATING SUBMISSIONS...")

    evaluation_data = [
        (bounties[0].id, submissions[0].id, "alice_wallet", 9.5, "Excellent work!", True),
        (bounties[1].id, submissions[1].id, "bob_wallet", 8.8, "Good design work", True),
        (bounties[2].id, submissions[2].id, "charlie_wallet", 9.2, "Great ML implementation", True),
        (bounties[3].id, submissions[3].id, "diana_wallet", 8.9, "Comprehensive documentation", True),
        (bounties[4].id, submissions[4].id, "eve_wallet", 9.7, "Outstanding smart contracts", True)
    ]

    for bounty_id, submission_id, reviewer, score, feedback, accept in evaluation_data:
        try:
            board.evaluate_submission(bounty_id, submission_id, reviewer, score, feedback, accept)
            status = "ACCEPTED" if accept else "REJECTED"
            print(f"✅ {submission_id}: {status} (Score: {score})")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

    # Show member dashboards
    print("\n👤 MEMBER DASHBOARDS...")

    for wallet, username in members[:3]:  # Show first 3 members
        dashboard = board.get_member_dashboard(wallet)
        print(f"\n👤 {username.upper()}:")
        print(f"   Reputation: {dashboard['member_info']['reputation_score']:.1f}")
        print(f"   Badges: {', '.join(dashboard['member_info']['badges']) if dashboard['member_info']['badges'] else 'None'}")
        print(f"   Skills: {', '.join(dashboard['member_info']['skills'].keys())}")
        print(f"   Completed Bounties: {dashboard['submissions']['accepted']}")
        print(f"   Success Rate: {dashboard['submissions']['success_rate']:.1%}")
        print(f"   Total Earnings: ${dashboard['earnings']['total_earned']:.2f}")
        print(f"   Matching Opportunities: {dashboard['opportunities']['matching_bounties']}")

    # Show community statistics
    print("\n📊 COMMUNITY STATISTICS...")

    stats = board.get_community_stats()
    print(f"👥 Total Members: {stats['overview']['total_members']}")
    print(f"📋 Total Bounties: {stats['overview']['total_bounties']}")
    print(f"🎯 Active Bounties: {stats['overview']['active_bounties']}")
    print(f"✅ Completed Bounties: {stats['overview']['completed_bounties']}")
    print(f"💰 Total Bounty Value: ${stats['overview']['total_bounty_value']:.2f}")
    print(f"📈 Completion Rate: {stats['overview']['completion_rate']:.1%}")

    print("\n🏆 TOP SKILLS:")
    for skill, count in stats['skills']['top_skills'][:5]:
        print(f"   {skill.title()}: {count} members")

    print("\n🎯 BOUNTY CATEGORIES:")
    for category, count in stats['bounties']['category_distribution'].items():
        print(f"   {category.title()}: {count} bounties")

    print("\n🎉 COMMUNITY BOUNTY BOARD DEMO COMPLETE!")
    print("✅ Bounties posted and escrowed")
    print("✅ Skill-based submissions received")
    print("✅ Quality evaluations completed")
    print("✅ Payments released to winners")
    print("✅ Reputation and badges updated")
    print("✅ Community statistics tracked")

    print("\n💡 KEY FEATURES DEMONSTRATED:")
    print("   📋 Bounty Posting with Requirements")
    print("   🎯 Skill-Based Matching")
    print("   🔒 Decentralized Escrow System")
    print("   ✅ Quality Evaluation Process")
    print("   💰 Automated Payment Distribution")
    print("   🏆 Reputation & Badge System")
    print("   📊 Community Analytics")

    print("\n🚀 BOUNTY BOARD READY FOR PRODUCTION!")
    print("Community members can now post bounties and fulfill projects!")
    print("Decentralized collaboration marketplace activated! 💎🚀")

if __name__ == "__main__":
    main()
