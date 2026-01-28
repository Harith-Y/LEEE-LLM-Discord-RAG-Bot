"""
Rate limiting to prevent abuse and ensure fair usage
"""
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Tuple, Dict
import asyncio
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for user requests
    """
    
    def __init__(self, max_requests: int = 5, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
        logger.info(f"RateLimiter initialized - {max_requests} requests per {window_seconds}s")
    
    async def is_allowed(self, user_id: str) -> Tuple[bool, int]:
        """
        Check if user is within rate limit
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Tuple of (is_allowed, seconds_until_reset)
            - is_allowed: True if request is allowed
            - seconds_until_reset: Seconds until rate limit resets (0 if allowed)
        """
        async with self._lock:
            now = datetime.now()
            user_requests = self.requests[user_id]
            
            # Remove requests outside the time window
            while user_requests and (now - user_requests[0]) > self.window:
                user_requests.popleft()
            
            # Check if user has exceeded rate limit
            if len(user_requests) >= self.max_requests:
                # Calculate time until oldest request expires
                oldest_request = user_requests[0]
                reset_time = oldest_request + self.window
                seconds_until_reset = int((reset_time - now).total_seconds()) + 1
                
                logger.warning(
                    f"Rate limit exceeded for user {user_id}: "
                    f"{len(user_requests)}/{self.max_requests} requests"
                )
                return False, seconds_until_reset
            
            # Allow request and record timestamp
            user_requests.append(now)
            remaining = self.max_requests - len(user_requests)
            logger.debug(
                f"Rate limit check passed for user {user_id}: "
                f"{len(user_requests)}/{self.max_requests} requests ({remaining} remaining)"
            )
            return True, 0
    
    async def get_user_stats(self, user_id: str) -> Dict[str, int]:
        """
        Get rate limit statistics for a user
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary with user statistics
        """
        async with self._lock:
            now = datetime.now()
            user_requests = self.requests[user_id]
            
            # Clean old requests
            while user_requests and (now - user_requests[0]) > self.window:
                user_requests.popleft()
            
            current_requests = len(user_requests)
            remaining = max(0, self.max_requests - current_requests)
            
            return {
                'current_requests': current_requests,
                'max_requests': self.max_requests,
                'remaining_requests': remaining,
                'window_seconds': self.window_seconds,
            }
    
    async def reset_user(self, user_id: str) -> None:
        """
        Reset rate limit for a specific user
        
        Args:
            user_id: Unique user identifier
        """
        async with self._lock:
            if user_id in self.requests:
                del self.requests[user_id]
                logger.info(f"Rate limit reset for user {user_id}")
    
    async def reset_all(self) -> None:
        """Reset rate limits for all users"""
        async with self._lock:
            self.requests.clear()
            logger.info("All rate limits reset")
    
    async def get_total_users(self) -> int:
        """Get total number of users being tracked"""
        async with self._lock:
            return len(self.requests)
    
    async def cleanup_inactive_users(self, inactive_minutes: int = 60) -> int:
        """
        Clean up rate limit data for users who haven't made requests recently
        
        Args:
            inactive_minutes: Minutes of inactivity before cleanup
            
        Returns:
            Number of users cleaned up
        """
        async with self._lock:
            now = datetime.now()
            inactive_threshold = timedelta(minutes=inactive_minutes)
            users_to_remove = []
            
            for user_id, user_requests in self.requests.items():
                if not user_requests or (now - user_requests[-1]) > inactive_threshold:
                    users_to_remove.append(user_id)
            
            for user_id in users_to_remove:
                del self.requests[user_id]
            
            if users_to_remove:
                logger.info(f"Cleaned up rate limit data for {len(users_to_remove)} inactive users")
            
            return len(users_to_remove)


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that can adapt limits based on server load or user behavior
    """
    
    def __init__(self, max_requests: int = 5, window_seconds: int = 60, 
                 trusted_multiplier: float = 2.0):
        """
        Initialize adaptive rate limiter
        
        Args:
            max_requests: Maximum requests for normal users
            window_seconds: Time window in seconds
            trusted_multiplier: Multiplier for trusted users
        """
        super().__init__(max_requests, window_seconds)
        self.trusted_multiplier = trusted_multiplier
        self.trusted_users = set()
    
    async def add_trusted_user(self, user_id: str) -> None:
        """
        Mark a user as trusted (higher rate limits)
        
        Args:
            user_id: User identifier
        """
        async with self._lock:
            self.trusted_users.add(user_id)
            logger.info(f"User {user_id} added to trusted list")
    
    async def remove_trusted_user(self, user_id: str) -> None:
        """
        Remove a user from trusted list
        
        Args:
            user_id: User identifier
        """
        async with self._lock:
            self.trusted_users.discard(user_id)
            logger.info(f"User {user_id} removed from trusted list")
    
    async def is_allowed(self, user_id: str) -> Tuple[bool, int]:
        """
        Check if user is within rate limit (with trusted user consideration)
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Tuple of (is_allowed, seconds_until_reset)
        """
        # Temporarily increase limit for trusted users
        original_max = self.max_requests
        
        async with self._lock:
            is_trusted = user_id in self.trusted_users
        
        if is_trusted:
            self.max_requests = int(original_max * self.trusted_multiplier)
        
        result = await super().is_allowed(user_id)
        
        # Restore original limit
        self.max_requests = original_max
        
        return result


# Global rate limiter instance
rate_limiter: RateLimiter = None


def get_rate_limiter(max_requests: int = 5, window_seconds: int = 60, 
                     adaptive: bool = False) -> RateLimiter:
    """
    Get or create global rate limiter instance
    
    Args:
        max_requests: Maximum requests allowed in time window
        window_seconds: Time window in seconds
        adaptive: Whether to use adaptive rate limiter
        
    Returns:
        RateLimiter instance
    """
    global rate_limiter
    if rate_limiter is None:
        if adaptive:
            rate_limiter = AdaptiveRateLimiter(max_requests, window_seconds)
        else:
            rate_limiter = RateLimiter(max_requests, window_seconds)
    return rate_limiter
