import Link from "next/link";
import { ArrowRight, ShieldCheck, Brain, Zap, Activity } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col font-sans bg-[#fafafa] selection:bg-indigo-500/30">
      {/* Header */}
      <header className="sticky top-0 w-full z-50 bg-white/70 backdrop-blur-lg border-b border-slate-200/50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex justify-between items-center">
          <div className="font-bold text-xl text-slate-900 tracking-tight flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span>Pulih.id</span>
          </div>
          <Link 
            href="/tes" 
            className="bg-slate-900 hover:bg-slate-800 text-white px-6 py-2.5 rounded-full font-medium transition-all text-sm shadow-sm hover:shadow-md"
          >
            Mulai Tes
          </Link>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative pt-24 pb-32 lg:pt-36 lg:pb-40 overflow-hidden flex-grow flex items-center justify-center">
        {/* Background Gradients */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full overflow-hidden -z-10 pointer-events-none">
          <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-indigo-400/20 rounded-full blur-[120px] mix-blend-multiply" />
          <div className="absolute top-[20%] right-[10%] w-[400px] h-[400px] bg-cyan-400/20 rounded-full blur-[100px] mix-blend-multiply" />
        </div>
        
        <div className="max-w-5xl mx-auto px-6 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-600 text-sm font-medium mb-8">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            Platform Skrining Kesehatan Mental
          </div>
          
          <h1 className="text-5xl md:text-7xl lg:text-[5.5rem] font-extrabold text-slate-900 tracking-tighter mb-8 leading-[1.1]">
            Kenali Kondisi <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-cyan-500">Mentalmu.</span>
          </h1>
          <p className="text-lg md:text-xl text-slate-600 max-w-2xl mx-auto mb-12 font-light leading-relaxed">
            Asesmen mandiri untuk mengenali tingkat depresi, kecemasan, dan stres dalam hidupmu. Cepat, aman, dan sangat mudah digunakan.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link 
              href="/tes" 
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-slate-900 text-white px-8 py-4 rounded-full font-semibold text-base hover:bg-slate-800 hover:scale-105 transition-all duration-300 shadow-xl shadow-slate-900/20"
            >
              Mulai Evaluasi Sekarang
              <ArrowRight className="w-5 h-5" />
            </Link>
            <a 
              href="#metode" 
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-white text-slate-700 px-8 py-4 rounded-full font-semibold text-base border border-slate-200 hover:bg-slate-50 transition-all duration-300"
            >
              Pelajari Lebih Lanjut
            </a>
          </div>
        </div>
      </section>

      {/* Stats / Trust Section */}
      <section className="border-y border-slate-200/60 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-10">
          <div className="flex flex-col md:flex-row items-center justify-center gap-10 md:gap-24">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600">
                <Activity className="w-6 h-6" />
              </div>
              <div>
                <div className="text-2xl font-bold text-slate-900">Holistik</div>
                <div className="text-sm text-slate-500 font-medium">Evaluasi Menyeluruh</div>
              </div>
            </div>
            <div className="hidden md:block w-px h-12 bg-slate-200"></div>
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600">
                <ShieldCheck className="w-6 h-6" />
              </div>
              <div>
                <div className="text-2xl font-bold text-slate-900">Aman</div>
                <div className="text-sm text-slate-500 font-medium">Privasi Terjamin</div>
              </div>
            </div>
            <div className="hidden md:block w-px h-12 bg-slate-200"></div>
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-cyan-50 rounded-2xl flex items-center justify-center text-cyan-600">
                <Zap className="w-6 h-6" />
              </div>
              <div>
                <div className="text-2xl font-bold text-slate-900">Cepat</div>
                <div className="text-sm text-slate-500 font-medium">Hasil Seketika</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Tests Section (Multi-label focus) */}
      <section id="metode" className="py-32 bg-[#fafafa]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-20">
            <h2 className="text-3xl md:text-5xl font-bold text-slate-900 mb-6 tracking-tight">Apa yang Dievaluasi?</h2>
            <p className="text-slate-600 max-w-3xl mx-auto text-lg leading-relaxed">
              Sistem kami mengevaluasi tiga dimensi psikologis utama depresi, kecemasan, dan stres menggunakan instrumen <strong className="text-indigo-600">DASS-42</strong> yang telah tervalidasi secara klinis,
              memberikan gambaran kondisi kesehatan mental secara lebih menyeluruh, akurat, dan mudah dipahami.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {/* Depresi Card */}
            <div className="group bg-white p-8 rounded-[2rem] border border-slate-200/60 shadow-sm hover:shadow-xl transition-all duration-500 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50 rounded-bl-[100px] -z-10 transition-all duration-500 group-hover:bg-indigo-100/50"></div>
              <div className="w-16 h-16 bg-white border border-slate-100 shadow-sm rounded-2xl flex items-center justify-center text-3xl mb-8">😔</div>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">Depresi</h3>
              <p className="text-slate-600 leading-relaxed min-h-[100px]">
                Mengukur tingkat keputusasaan, devaluasi kehidupan, kurangnya minat, serta kurangnya motivasi dalam aktivitas sehari-hari.
              </p>
            </div>

            {/* Kecemasan Card */}
            <div className="group bg-white p-8 rounded-[2rem] border border-slate-200/60 shadow-sm hover:shadow-xl transition-all duration-500 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-50 rounded-bl-[100px] -z-10 transition-all duration-500 group-hover:bg-cyan-100/50"></div>
              <div className="w-16 h-16 bg-white border border-slate-100 shadow-sm rounded-2xl flex items-center justify-center text-3xl mb-8">😰</div>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">Kecemasan</h3>
              <p className="text-slate-600 leading-relaxed min-h-[100px]">
                Mengevaluasi respons otonomik, efek otot skeletal, kecemasan situasional, dan pengalaman subjektif terkait kepanikan berlebih.
              </p>
            </div>

            {/* Stres Card */}
            <div className="group bg-white p-8 rounded-[2rem] border border-slate-200/60 shadow-sm hover:shadow-xl transition-all duration-500 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-rose-50 rounded-bl-[100px] -z-10 transition-all duration-500 group-hover:bg-rose-100/50"></div>
              <div className="w-16 h-16 bg-white border border-slate-100 shadow-sm rounded-2xl flex items-center justify-center text-3xl mb-8">😣</div>
              <h3 className="text-2xl font-bold text-slate-900 mb-4">Stres</h3>
              <p className="text-slate-600 leading-relaxed min-h-[100px]">
                Menilai tingkat kesulitan rileks, gairah saraf, mudah marah, agitasi, ketidaksabaran, dan reaktivitas berlebihan terhadap situasi.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-32 bg-white border-t border-slate-100">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl md:text-5xl font-bold text-slate-900 mb-6 tracking-tight">Kenapa Memilih Pulih.id?</h2>
              <p className="text-slate-600 text-lg leading-relaxed mb-10">
                Kami merancang pengalaman tes yang aman, nyaman, dan modern untuk membantu Anda memahami diri sendiri dengan lebih baik.
              </p>
              
              <div className="space-y-8">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center text-slate-700">
                    <ShieldCheck className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="text-xl font-bold text-slate-900 mb-2">Privasi Penuh</h4>
                    <p className="text-slate-600">Semua hasil tes diproses secara aman dan tidak disimpan di database manapun. Tes ini sepenuhnya anonim.</p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center text-slate-700">
                    <Brain className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="text-xl font-bold text-slate-900 mb-2">Analisis Pintar</h4>
                    <p className="text-slate-600">Menganalisis pola gaya hidup, kepribadian, dan demografi Anda untuk memberikan hasil yang relevan.</p>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center text-slate-700">
                    <Zap className="w-6 h-6" />
                  </div>
                  <div>
                    <h4 className="text-xl font-bold text-slate-900 mb-2">Hasil Interaktif</h4>
                    <p className="text-slate-600">Dapatkan persentase tingkat risiko secara instan dalam antarmuka visual yang mudah dipahami.</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="relative mt-10 lg:mt-0">
              <div className="absolute inset-0 bg-gradient-to-tr from-indigo-100 to-cyan-50 rounded-[3rem] transform rotate-3 scale-105 -z-10"></div>
              <div className="bg-white border border-slate-200/60 p-8 rounded-[3rem] shadow-2xl shadow-slate-200/50">
                <div className="space-y-6">
                  {/* Mockup UI Elements */}
                  <div className="h-4 w-32 bg-slate-100 rounded-full mb-8"></div>
                  
                  <div className="p-5 border border-slate-100 rounded-2xl flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="text-2xl">😔</div>
                      <div>
                        <div className="font-bold text-slate-900">Risiko Depresi</div>
                        <div className="text-sm text-slate-500">Probabilitas</div>
                      </div>
                    </div>
                    <div className="text-xl font-extrabold text-indigo-600">Low</div>
                  </div>

                  <div className="p-5 border border-slate-100 rounded-2xl flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="text-2xl">😰</div>
                      <div>
                        <div className="font-bold text-slate-900">Risiko Kecemasan</div>
                        <div className="text-sm text-slate-500">Probabilitas</div>
                      </div>
                    </div>
                    <div className="text-xl font-extrabold text-cyan-600">Med</div>
                  </div>

                  <div className="p-5 border border-slate-100 rounded-2xl flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="text-2xl">😣</div>
                      <div>
                        <div className="font-bold text-slate-900">Risiko Stres</div>
                        <div className="text-sm text-slate-500">Probabilitas</div>
                      </div>
                    </div>
                    <div className="text-xl font-extrabold text-rose-500">High</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 bg-slate-50 border-t border-slate-100">
        <div className="max-w-3xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4 tracking-tight">Pertanyaan yang Sering Diajukan</h2>
            <p className="text-slate-600 text-lg">Temukan jawaban untuk pertanyaan umum seputar tes di Pulih.id</p>
          </div>
          
          <div className="space-y-4">
            <details className="group bg-white rounded-2xl border border-slate-200 shadow-sm [&_summary::-webkit-details-marker]:hidden">
              <summary className="flex items-center justify-between p-6 cursor-pointer font-semibold text-slate-900">
                Apakah hasil tes ini bisa dijadikan diagnosis medis?
                <span className="transition duration-300 group-open:-rotate-180">
                  <svg fill="none" height="24" shapeRendering="geometricPrecision" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" viewBox="0 0 24 24" width="24"><path d="M6 9l6 6 6-6"></path></svg>
                </span>
              </summary>
              <div className="px-6 pb-6 text-slate-600 leading-relaxed">
                Tidak. Pulih.id adalah alat skrining mandiri yang bertujuan untuk meningkatkan kesadaran diri (self-awareness). Hasil dari platform ini bukan merupakan diagnosis medis yang pasti dan tidak dapat menggantikan konsultasi langsung dengan psikolog atau psikiater.
              </div>
            </details>
            
            <details className="group bg-white rounded-2xl border border-slate-200 shadow-sm [&_summary::-webkit-details-marker]:hidden">
              <summary className="flex items-center justify-between p-6 cursor-pointer font-semibold text-slate-900">
                Apakah data jawaban saya aman?
                <span className="transition duration-300 group-open:-rotate-180">
                  <svg fill="none" height="24" shapeRendering="geometricPrecision" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" viewBox="0 0 24 24" width="24"><path d="M6 9l6 6 6-6"></path></svg>
                </span>
              </summary>
              <div className="px-6 pb-6 text-slate-600 leading-relaxed">
                Tentu saja. Privasi Anda adalah prioritas kami. Semua data jawaban diproses secara aman dan tidak disimpan secara permanen di database kami. Anda tidak perlu memasukkan nama atau identitas pribadi.
              </div>
            </details>

            <details className="group bg-white rounded-2xl border border-slate-200 shadow-sm [&_summary::-webkit-details-marker]:hidden">
              <summary className="flex items-center justify-between p-6 cursor-pointer font-semibold text-slate-900">
                Berapa lama waktu yang dibutuhkan untuk tes?
                <span className="transition duration-300 group-open:-rotate-180">
                  <svg fill="none" height="24" shapeRendering="geometricPrecision" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" viewBox="0 0 24 24" width="24"><path d="M6 9l6 6 6-6"></path></svg>
                </span>
              </summary>
              <div className="px-6 pb-6 text-slate-600 leading-relaxed">
                Tes ini sangat praktis dan biasanya hanya memakan waktu sekitar 3 hingga 5 menit. Kami menyarankan Anda untuk menjawab setiap pertanyaan secara jujur agar hasil yang didapatkan lebih relevan.
              </div>
            </details>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-16 mt-auto">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-12 mb-16">
            <div className="lg:col-span-2">
              <div className="font-bold text-2xl text-slate-900 tracking-tight flex items-center gap-2 mb-6">
                <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <span>Pulih.id</span>
              </div>
              <p className="text-slate-500 max-w-md leading-relaxed mb-8">
                Platform skrining kesehatan mental mandiri yang aman dan mudah digunakan untuk membantu Anda memahami kondisi psikologis Anda.
              </p>
              <div className="bg-rose-50 border border-rose-100 rounded-2xl p-5 max-w-md text-rose-700 text-sm">
                <strong className="block mb-1 text-rose-800">Disclaimer Medis</strong>
                Platform ini bukan pengganti diagnosis klinis. Hubungi <strong>119</strong> jika Anda memerlukan bantuan profesional darurat.
              </div>
            </div>
            
            <div>
              <h4 className="font-bold text-slate-900 mb-6">Platform</h4>
              <ul className="space-y-4">
                <li><a href="#" className="text-slate-500 hover:text-indigo-600 transition-colors">Beranda</a></li>
                <li><Link href="/tes" className="text-slate-500 hover:text-indigo-600 transition-colors">Mulai Tes</Link></li>
                <li><a href="#metode" className="text-slate-500 hover:text-indigo-600 transition-colors">Apa yang Dievaluasi</a></li>
                <li><a href="#" className="text-slate-500 hover:text-indigo-600 transition-colors">Kebijakan Privasi</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-bold text-slate-900 mb-6">Informasi</h4>
              <ul className="space-y-4">
                <li><span className="text-slate-500">Tentang Kami</span></li>
                <li><span className="text-slate-500">Pusat Bantuan</span></li>
                <li><span className="text-slate-500">Hubungi Kami</span></li>
              </ul>
            </div>
          </div>
          
          <div className="pt-8 border-t border-slate-100 text-sm text-slate-400 font-medium flex flex-col md:flex-row justify-between items-center gap-4">
            <div>&copy; {new Date().getFullYear()} Pulih.id. Hak Cipta Dilindungi.</div>
            <div className="flex gap-6">
              <a href="#" className="hover:text-slate-600 transition-colors">Syarat & Ketentuan</a>
              <a href="#" className="hover:text-slate-600 transition-colors">Hubungi Kami</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
