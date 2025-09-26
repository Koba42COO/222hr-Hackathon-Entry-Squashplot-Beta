import { TestBed } from '@angular/core/testing';
import { AuthService } from './auth.service';

describe('AuthService', () => {
  let service: AuthService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [AuthService]
    });
    service = TestBed.inject(AuthService);
    
    // Clear localStorage before each test
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start with no authenticated user', () => {
    service.currentUser$.subscribe(user => {
      expect(user).toBeNull();
    });

    service.isAuthenticated$.subscribe(isAuth => {
      expect(isAuth).toBeFalse();
    });
  });

  it('should login successfully with valid credentials', () => {
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    
    service.login(credentials).subscribe(result => {
      expect(result.success).toBeTrue();
      expect(result.user).toBeDefined();
      expect(result.token).toBeDefined();
    });

    service.isAuthenticated$.subscribe(isAuth => {
      expect(isAuth).toBeTrue();
    });

    service.currentUser$.subscribe(user => {
      expect(user).toBeDefined();
      expect(user?.username).toBe('admin');
    });
  });

  it('should fail login with invalid credentials', () => {
    const credentials = { username: 'wrong', email: 'wrong@test.com', password: 'wrong' };
    
    service.login(credentials).subscribe(result => {
      expect(result.success).toBeFalse();
      expect(result.error).toBeDefined();
    });

    service.isAuthenticated$.subscribe(isAuth => {
      expect(isAuth).toBeFalse();
    });
  });

  it('should logout successfully', () => {
    // First login
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    // Then logout
    service.logout();

    service.isAuthenticated$.subscribe(isAuth => {
      expect(isAuth).toBeFalse();
    });

    service.currentUser$.subscribe(user => {
      expect(user).toBeNull();
    });

    expect(service.token).toBeNull();
  });

  it('should persist authentication state in localStorage', () => {
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    
    service.login(credentials).subscribe(result => {
      expect(localStorage.getItem('chaios_token')).toBe(result.token);
      expect(localStorage.getItem('chaios_user')).toBeDefined();
    });
  });

  it('should restore authentication state from localStorage', () => {
    // Manually set localStorage as if user was previously logged in
    const mockUser = {
      id: '1',
      username: 'admin',
      email: 'admin@chaios.ai',
      role: 'admin' as const,
      permissions: ['read', 'write', 'admin'] as const,
      lastLogin: new Date(),
      isActive: true
    };
    const mockToken = 'mock-jwt-token';

    localStorage.setItem('chaios_token', mockToken);
    localStorage.setItem('chaios_user', JSON.stringify(mockUser));

    // Create new service instance to trigger initialization
    const newService = new AuthService();

    newService.isAuthenticated$.subscribe(isAuth => {
      expect(isAuth).toBeTrue();
    });

    newService.currentUser$.subscribe(user => {
      expect(user).toEqual(jasmine.objectContaining({
        id: '1',
        username: 'admin',
        email: 'admin@chaios.ai'
      }));
    });

    expect(newService.token).toBe(mockToken);
  });

  it('should register new user successfully', () => {
    const userData = {
      username: 'newuser',
      email: 'newuser@test.com',
      password: 'password123',
      confirmPassword: 'password123'
    };

    service.register(userData).subscribe(result => {
      expect(result.success).toBeTrue();
      expect(result.user).toBeDefined();
      expect(result.user?.username).toBe('newuser');
    });
  });

  it('should fail registration with mismatched passwords', () => {
    const userData = {
      username: 'newuser',
      email: 'newuser@test.com',
      password: 'password123',
      confirmPassword: 'different'
    };

    service.register(userData).subscribe(result => {
      expect(result.success).toBeFalse();
      expect(result.error).toBe('Passwords do not match');
    });
  });

  it('should fail registration with existing username', () => {
    const userData = {
      username: 'admin', // Already exists
      email: 'admin2@test.com',
      password: 'password123',
      confirmPassword: 'password123'
    };

    service.register(userData).subscribe(result => {
      expect(result.success).toBeFalse();
      expect(result.error).toBe('Username already exists');
    });
  });

  it('should update user profile', () => {
    // First login
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    const updates = {
      email: 'newemail@chaios.ai',
      firstName: 'John',
      lastName: 'Doe'
    };

    service.updateProfile(updates).subscribe(result => {
      expect(result.success).toBeTrue();
      expect(result.user?.email).toBe('newemail@chaios.ai');
    });

    service.currentUser$.subscribe(user => {
      expect(user?.email).toBe('newemail@chaios.ai');
    });
  });

  it('should change password successfully', () => {
    // First login
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    service.changePassword('admin123', 'newpassword123').subscribe(result => {
      expect(result.success).toBeTrue();
    });
  });

  it('should fail password change with wrong current password', () => {
    // First login
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    service.changePassword('wrongpassword', 'newpassword123').subscribe(result => {
      expect(result.success).toBeFalse();
      expect(result.error).toBe('Current password is incorrect');
    });
  });

  it('should handle token expiration', () => {
    // Login first
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    // Simulate token expiration by calling logout
    service.logout();

    service.isAuthenticated$.subscribe(isAuth => {
      expect(isAuth).toBeFalse();
    });

    service.currentUser$.subscribe(user => {
      expect(user).toBeNull();
    });

    expect(service.token).toBeNull();
  });

  it('should check if user has required permissions', () => {
    // Login first
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    expect(service.hasPermission('read')).toBeTrue();
    expect(service.hasPermission('write')).toBeTrue();
    expect(service.hasPermission('admin')).toBeTrue();
    expect(service.hasPermission('nonexistent' as any)).toBeFalse();
  });

  it('should check if user has required role', () => {
    // Login first
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    expect(service.hasRole('admin')).toBeTrue();
    expect(service.hasRole('user')).toBeFalse();
  });

  it('should refresh token', () => {
    // Login first
    const credentials = { username: 'admin', email: 'admin@chaios.ai', password: 'admin123' };
    service.login(credentials).subscribe();

    const oldToken = service.token;

    service.refreshToken().subscribe(result => {
      expect(result.user).toBeDefined();
      expect(result.user?.id).toBeDefined();
      expect(service.token).not.toBe(oldToken);
    });
  });
});
