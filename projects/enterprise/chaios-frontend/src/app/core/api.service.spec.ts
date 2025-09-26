import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

describe('ApiService', () => {
  let service: ApiService;
  let httpMock: HttpTestingController;
  let authServiceSpy: jasmine.SpyObj<AuthService>;

  beforeEach(() => {
    const spy = jasmine.createSpyObj('AuthService', ['getToken']);

    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [
        ApiService,
        { provide: AuthService, useValue: spy }
      ]
    });

    service = TestBed.inject(ApiService);
    httpMock = TestBed.inject(HttpTestingController);
    authServiceSpy = TestBed.inject(AuthService) as jasmine.SpyObj<AuthService>;
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should send chat message', () => {
    const mockResponse = { 
      success: true, 
      response: 'Test response',
      conversational_response: 'Conversational test response'
    };
    const message = 'Test message';

    authServiceSpy.getToken.and.returnValue('mock-token');

    service.sendChatMessage(message).subscribe(response => {
      expect(response).toEqual(mockResponse);
    });

    const req = httpMock.expectOne('http://localhost:8000/consciousness/process');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({
      query: message,
      user_id: 'frontend_user',
      session_id: jasmine.any(String),
      enhancement_level: 3,
      use_system_tools: true,
      permissions: jasmine.any(Array)
    });
    expect(req.request.headers.get('Authorization')).toBe('Bearer mock-token');

    req.flush(mockResponse);
  });

  it('should execute system tool', () => {
    const mockResponse = { 
      success: true, 
      result: { data: 'test result' },
      execution_time: 1.5
    };
    const toolName = 'test_tool';
    const parameters = { param1: 'value1' };

    authServiceSpy.getToken.and.returnValue('mock-token');

    service.executeSystemTool(toolName, parameters).subscribe(response => {
      expect(response).toEqual(mockResponse);
    });

    const req = httpMock.expectOne('http://localhost:8000/system-tools/execute');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({
      tool_name: toolName,
      parameters: parameters,
      user_id: 'frontend_user',
      session_id: jasmine.any(String),
      permissions: jasmine.any(Array)
    });

    req.flush(mockResponse);
  });

  it('should get system tools', () => {
    const mockTools = [
      { name: 'tool1', description: 'Tool 1', category: 'general' },
      { name: 'tool2', description: 'Tool 2', category: 'ai_ml' }
    ];

    authServiceSpy.getToken.and.returnValue('mock-token');

    service.getSystemTools().subscribe(tools => {
      expect(tools).toEqual(mockTools);
    });

    const req = httpMock.expectOne('http://localhost:8000/system-tools/list');
    expect(req.request.method).toBe('POST');
    expect(req.request.headers.get('Authorization')).toBe('Bearer mock-token');

    req.flush(mockTools);
  });

  it('should batch execute system tools', () => {
    const mockResponse = {
      success: true,
      results: [
        { tool_name: 'tool1', success: true, result: 'result1' },
        { tool_name: 'tool2', success: true, result: 'result2' }
      ]
    };
    const requests = [
      { tool_name: 'tool1', parameters: {} },
      { tool_name: 'tool2', parameters: {} }
    ];

    authServiceSpy.getToken.and.returnValue('mock-token');

    service.batchExecuteSystemTools(requests).subscribe(response => {
      expect(response).toEqual(mockResponse);
    });

    const req = httpMock.expectOne('http://localhost:8000/system-tools/batch-execute');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({
      requests: requests,
      user_id: 'frontend_user',
      session_id: jasmine.any(String),
      permissions: jasmine.any(Array)
    });

    req.flush(mockResponse);
  });

  it('should handle HTTP errors gracefully', () => {
    const message = 'Test message';
    
    authServiceSpy.getToken.and.returnValue('mock-token');

    service.sendChatMessage(message).subscribe({
      next: () => fail('should have failed'),
      error: (error) => {
        expect(error.status).toBe(500);
      }
    });

    const req = httpMock.expectOne('http://localhost:8000/consciousness/process');
    req.flush('Server error', { status: 500, statusText: 'Internal Server Error' });
  });

  it('should include authorization header when token is available', () => {
    const mockResponse = { success: true, response: 'Test' };
    const token = 'test-jwt-token';
    
    authServiceSpy.getToken.and.returnValue(token);

    service.sendChatMessage('test').subscribe();

    const req = httpMock.expectOne('http://localhost:8000/consciousness/process');
    expect(req.request.headers.get('Authorization')).toBe(`Bearer ${token}`);
    
    req.flush(mockResponse);
  });

  it('should work without authorization header when no token', () => {
    const mockResponse = { success: true, response: 'Test' };
    
    authServiceSpy.getToken.and.returnValue(null);

    service.sendChatMessage('test').subscribe();

    const req = httpMock.expectOne('http://localhost:8000/consciousness/process');
    expect(req.request.headers.has('Authorization')).toBeFalse();
    
    req.flush(mockResponse);
  });
});
