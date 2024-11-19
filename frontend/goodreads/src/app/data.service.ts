import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  private apiUrl = 'http://localhost:8000/api/text';
  constructor(private http: HttpClient) { }

  getTextFromFastAPI(): Observable<any> {
    return this.http.get<any>(this.apiUrl);
  }
}
