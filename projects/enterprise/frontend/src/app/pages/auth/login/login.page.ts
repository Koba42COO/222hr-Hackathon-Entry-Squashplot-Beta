import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';
import { RouterModule } from '@angular/router';
import { ApiService } from '../../../core/services/api';

@Component({
  selector: 'app-login',
  templateUrl: './login.page.html',
  styleUrls: ['./login.page.scss'],
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    ReactiveFormsModule,
    RouterModule,
  ],
})
export class LoginPage implements OnInit {
  loginForm: FormGroup;
  loading = false;
  error: string | null = null;

  constructor(
    private formBuilder: FormBuilder,
    private apiService: ApiService,
    private router: Router
  ) {
    this.loginForm = this.formBuilder.group({
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]]
    });
  }

  ngOnInit() {}

  onSubmit() {
    if (this.loginForm.valid && !this.loading) {
      this.loading = true;
      this.error = null;
      
      const { email, password } = this.loginForm.value;
      
      this.apiService.login(email, password).subscribe({
        next: (response) => {
          if (response.success && response.data?.token) {
            localStorage.setItem('auth_token', response.data.token);
            localStorage.setItem('user_data', JSON.stringify(response.data.user));
            
            // Redirect based on user role
            if (response.data.user?.role === 'admin') {
              this.router.navigate(['/admin']);
            } else {
              this.router.navigate(['/llm-convo']);
            }
          } else {
            this.error = response.error || 'Login failed';
          }
          this.loading = false;
        },
        error: (error) => {
          this.error = error;
          this.loading = false;
        }
      });
    }
  }

  clearError() {
    this.error = null;
  }

  quickAdminLogin() {
    // Fill form with admin credentials
    this.loginForm.patchValue({
      email: 'admin@koba42corp.com',
      password: 'admin123'
    });
    
    // Submit the form
    this.onSubmit();
  }
}
