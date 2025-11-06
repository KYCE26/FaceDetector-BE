import uvicorn
import firebase_admin
import numpy as np
from fastapi import FastAPI, HTTPException
from firebase_admin import credentials, firestore
from pydantic import BaseModel
from typing import List
import os     # <-- Ditambahkan untuk deploy
import json   # <-- Ditambahkan untuk deploy

# --- Model Data (Tidak berubah) ---
class UserRegister(BaseModel):
    user_id: str
    name: str
    embedding: List[float]

class RecognizeRequest(BaseModel):
    embedding: List[float]

# =========================================================
# ▼▼▼ BLOK INISIALISASI BARU (SIAP DEPLOY) ▼▼▼
# =========================================================
try:
    # 1. Coba mode DEPLOY (Railway)
    # Ambil JSON string dari Environment Variable
    creds_json_string = os.getenv("FIREBASE_CREDS_JSON")
    
    if creds_json_string:
        # Jika ada, ubah JSON string menjadi dictionary
        creds_dict = json.loads(creds_json_string)
        cred = credentials.Certificate(creds_dict)
    else:
        # 2. Jika GAGAL (var tidak ada), coba mode LOKAL
        cred = credentials.Certificate("serviceAccountKey.json")
    
    # Inisialisasi app
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

except FileNotFoundError:
    print("FATAL ERROR: Gagal mode Lokal. 'serviceAccountKey.json' tidak ditemukan.")
    exit()
except json.JSONDecodeError:
    print("FATAL ERROR: Gagal mode Deploy. FIREBASE_CREDS_JSON bukan JSON valid.")
    exit()
except Exception as e:
    print(f"FATAL ERROR saat inisialisasi Firebase: {e}")
    exit()
        
db = firestore.client()
app = FastAPI(title="API Presensi Wajah V3 (Sub-collections)")
# =========================================================
# ▲▲▲ AKHIR BLOK INISIALISASI BARU ▲▲▲
# =========================================================

# --- Helper Function (Tidak berubah) ---
def calculate_cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return similarity

# --- API Endpoints (Tidak berubah) ---

@app.post("/register")
async def register_face(user: UserRegister):
    """
    Endpoint mendaftar.
    Menyimpan data user (nama) di dokumen, 
    dan embedding di sub-collection 'embeddings'.
    """
    try:
        if len(user.embedding) != 192:
            raise HTTPException(400, "Embedding harus 192-dimensi")
            
        # Referensi ke dokumen user
        user_doc_ref = db.collection("users").document(user.user_id)
        
        # Update/buat data user (nama)
        # 'merge=True' agar tidak menimpa field lain jika dokumen sudah ada
        user_doc_ref.set({
            "name": user.name
        }, merge=True)
        
        # Tambahkan embedding baru ke sub-collection 'embeddings'
        # .add() akan membuat dokumen baru dengan ID acak
        embedding_ref = user_doc_ref.collection("embeddings").add({
            "vector": user.embedding,
            "created_at": firestore.SERVER_TIMESTAMP # Opsional, tapi data bagus
        })
        
        return {
            "status": "success", 
            "user_id": user.user_id, 
            "name": user.name,
            # Kita kirim ID dokumen embedding baru sebagai bukti
            "new_embedding_id": embedding_ref[1].id 
        }
        
    except Exception as e:
        # Menangkap error spesifik Firestore jika terjadi
        if "Nested arrays are not allowed" in str(e):
             raise HTTPException(400, "Firestore Error: Nested arrays are not allowed. Periksa data Anda.")
        raise HTTPException(500, f"Gagal mendaftar: {str(e)}")


@app.post("/recognize")
async def recognize_face(req: RecognizeRequest):
    """
    Endpoint mengenali.
    Membaca semua users, lalu semua embeddings di sub-collection mereka.
    """
    try:
        if len(req.embedding) != 192:
            raise HTTPException(400, "Embedding harus 192-dimensi")

        current_embedding = req.embedding
        threshold = 0.80  # Sesuai permintaan Anda, 80%

        best_match_name = "Tidak Dikenali"
        best_match_id = None
        best_overall_similarity = 0.0

        # 1. Ambil semua user
        users_ref = db.collection("users").stream()

        for user in users_ref:
            user_data = user.to_dict()
            user_id = user.id
            user_name = user_data.get("name", "Nama tidak ada")
            
            # 2. Ambil semua embedding di sub-collection user ini
            embeddings_ref = user.reference.collection("embeddings").stream()
            
            best_similarity_for_this_user = 0.0

            for embedding_doc in embeddings_ref:
                embedding_data = embedding_doc.to_dict()
                registered_embedding = embedding_data.get("vector")
                
                if not registered_embedding:
                    continue # Lewati jika dokumen embedding rusak

                similarity = calculate_cosine_similarity(current_embedding, registered_embedding)
                
                if similarity > best_similarity_for_this_user:
                    best_similarity_for_this_user = similarity

            # 3. Bandingkan skor terbaik user ini dengan skor global
            if best_similarity_for_this_user > best_overall_similarity:
                best_overall_similarity = best_similarity_for_this_user
                best_match_name = user_name
                best_match_id = user_id

        # 4. Cek hasil akhir
        if best_overall_similarity >= threshold:
            return {
                "status": "success",
                "user_id": best_match_id,
                "name": best_match_name,
                "similarity": best_overall_similarity
            }
        else:
            # Ini adalah respons "tidak dikenali" yang normal
            raise HTTPException(404, detail=f"Wajah tidak dikenali. Kemiripan tertinggi: {best_overall_similarity:.2f}")

    except Exception as e:
        raise HTTPException(500, f"Gagal mengenali: {str(e)}")


if __name__ == "__main__":
    # Ini HANYA akan berjalan jika Anda menjalankan 'python main.py'
    # Ini akan diabaikan oleh Procfile di Railway
    uvicorn.run(app, host="0.0.0.0", port=8000)