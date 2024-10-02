import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  private apiUrl = 'http://localhost:5000/api/text';
  constructor(private http: HttpClient) { }

  getTextFromFlask(): Observable<any> {
    return this.http.get<any>(this.apiUrl);
  }
}
