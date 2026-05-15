"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft, Brain, ChevronRight, ChevronLeft, CheckCircle, AlertTriangle } from "lucide-react";

// --- Data Pertanyaan Lengkap (Akan di-filter dinamis berdasarkan API) ---

const demografi = [
  { id: "age", label: "Usia", type: "number", min: 15, max: 24, defaultValue: 20 },
  { id: "gender", label: "Jenis Kelamin", type: "select", options: [{val: 1, label: "Laki-laki"}, {val: 2, label: "Perempuan"}] },
  { id: "engnat", label: "Bahasa Inggris sebagai Bahasa Utama", type: "select", options: [{val: 0, label: "Tidak"}, {val: 1, label: "Ya"}] },
  { id: "education", label: "Tingkat Pendidikan", type: "select", options: [{val: 1, label: "Di Bawah SMA"}, {val: 2, label: "SMA/Sederajat"}, {val: 3, label: "Sarjana (S1)"}, {val: 4, label: "Pascasarjana (S2/S3)"}] },
  { id: "urban", label: "Wilayah Tempat Tinggal", type: "select", options: [{val: 1, label: "Pedesaan"}, {val: 2, label: "Pinggiran Kota"}, {val: 3, label: "Perkotaan"}] },
  { id: "religion", label: "Agama", type: "select", options: [{val: 0, label: "Agnostik"}, {val: 1, label: "Ateis"}, {val: 2, label: "Buddha"}, {val: 3, label: "Kristen"}, {val: 4, label: "Hindu"}, {val: 5, label: "Yahudi"}, {val: 6, label: "Islam"}] },
  { id: "orientation", label: "Orientasi Seksual", type: "select", options: [{val: 0, label: "Heteroseksual"}, {val: 1, label: "Biseksual"}, {val: 2, label: "Homoseksual"}] },
  { id: "race", label: "Ras/Etnis", type: "select", options: [{val: 0, label: "Kulit Putih"}, {val: 1, label: "Kulit Hitam"}, {val: 2, label: "Asia"}, {val: 3, label: "Campuran"}, {val: 4, label: "Lainnya"}, {val: 5, label: "Latin"}] },
  { id: "voted", label: "Suara Politik", type: "select", options: [{val: 0, label: "Tidak"}, {val: 1, label: "Ya"}, {val: 2, label: "Abstain"}, {val: 3, label: "Tidak Berlaku"}] },
  { id: "married", label: "Status Pernikahan", type: "select", options: [{val: 0, label: "Belum Menikah"}, {val: 1, label: "Menikah"}, {val: 2, label: "Cerai"}, {val: 3, label: "Janda/Duda"}, {val: 4, label: "Pisah"}] },
  { id: "familysize", label: "Jumlah Anggota Keluarga", type: "number", min: 0, max: 10, defaultValue: 3 },
];

const tipi = [
  { id: "TIPI1", text: "Saya adalah orang yang riang, ekstrovert", desc: "(Sebaliknya: Pendiam/Tertutup)" },
  { id: "TIPI2", text: "Saya adalah orang yang kritis, suka berdebat", desc: "(Sebaliknya: Pemaaf/Lembut)" },
  { id: "TIPI3", text: "Saya adalah orang yang dapat diandalkan, teratur", desc: "(Sebaliknya: Ceroboh)" },
  { id: "TIPI4", text: "Saya mudah cemas, terguncang emosionalnya", desc: "(Sebaliknya: Tenang/Stabil)" },
  { id: "TIPI5", text: "Saya terbuka terhadap pengalaman baru, kreatif", desc: "(Sebaliknya: Konvensional)" },
  { id: "TIPI6", text: "Saya adalah orang yang pendiam, tertutup", desc: "(Sebaliknya: Ekstrovert/Sosial)" },
  { id: "TIPI7", text: "Saya adalah orang yang simpatik, hangat", desc: "(Sebaliknya: Kritis)" },
  { id: "TIPI8", text: "Saya adalah orang yang tidak terorganisir, ceroboh", desc: "(Sebaliknya: Rapi)" },
  { id: "TIPI9", text: "Saya adalah orang yang tenang, stabil emosional", desc: "(Sebaliknya: Mudah cemas)" },
  { id: "TIPI10", text: "Saya adalah orang yang konvensional", desc: "(Sebaliknya: Terbuka/Inovatif)" }
];

const vcl = [
  { id: "VCL1", text: "Konsumsi Alkohol" },
  { id: "VCL2", text: "Penggunaan Benzodiazepine" },
  { id: "VCL3", text: "Konsumsi Kafein" },
  { id: "VCL4", text: "Penggunaan Cannabis" },
  { id: "VCL5", text: "Penggunaan Kokain" },
  { id: "VCL6", text: "Penggunaan Heroin" },
  { id: "VCL7", text: "Penggunaan Amfetamin" },
  { id: "VCL8", text: "Penggunaan Ecstasy" },
  { id: "VCL9", text: "Penggunaan LSD" },
  { id: "VCL10", text: "Penggunaan Psilocybin (Jamur)" },
  { id: "VCL11", text: "Legal Highs" },
  { id: "VCL12", text: "Penggunaan Tembakau/Nikotin" },
  { id: "VCL13", text: "Penggunaan Metadon" },
  { id: "VCL14", text: "Penggunaan VSA" },
  { id: "VCL15", text: "Penggunaan PCP" },
  { id: "VCL16", text: "Penggunaan Crack" }
];

const tipiOptions = [
  { val: 1, label: "Sangat Tidak Setuju" },
  { val: 2, label: "Tidak Setuju" },
  { val: 3, label: "Agak Tidak Setuju" },
  { val: 4, label: "Netral" },
  { val: 5, label: "Agak Setuju" },
  { val: 6, label: "Setuju" },
  { val: 7, label: "Sangat Setuju" }
];

const vclOptions = [
  { val: 0, label: "Tidak Pernah" },
  { val: 1, label: "Lebih dari 10 Tahun" },
  { val: 2, label: "Kurang dari 10 Tahun" },
  { val: 3, label: "Satu Tahun Terakhir" },
  { val: 4, label: "Satu Bulan Terakhir" },
  { val: 5, label: "Satu Minggu Terakhir" },
  { val: 6, label: "Satu Hari Terakhir" }
];

const getExplanation = (condition: 'depression' | 'anxiety' | 'stress', risk: 'high' | 'medium' | 'low') => {
  const explanations = {
    depression: {
      high: { desc: "Skor menunjukkan kemungkinan besar mengalami gejala depresi seperti perasaan sedih berkepanjangan, kehilangan minat, atau gangguan tidur.", rec: "Sangat disarankan untuk segera berkonsultasi dengan profesional kesehatan mental." },
      medium: { desc: "Ada beberapa indikator yang menunjukkan potensi gejala depresi ringan hingga sedang.", rec: "Pertimbangkan untuk berbicara dengan konselor atau psikolog untuk evaluasi lebih lanjut." },
      low: { desc: "Skor menunjukkan kemungkinan rendah mengalami gejala depresi saat ini.", rec: "Tetap jaga kesehatan mental dengan pola hidup sehat dan dukungan sosial yang baik." }
    },
    anxiety: {
      high: { desc: "Skor menunjukkan kemungkinan besar mengalami gejala kecemasan berlebihan, seperti rasa khawatir yang tidak terkendali, jantung berdebar, atau kesulitan berkonsentrasi.", rec: "Sangat disarankan untuk segera mencari bantuan profesional." },
      medium: { desc: "Ada indikasi potensi gangguan kecemasan ringan.", rec: "Teknik relaksasi, mindfulness, dan konsultasi awal dengan profesional dapat membantu." },
      low: { desc: "Skor menunjukkan tingkat kecemasan yang normal.", rec: "Pertahankan kebiasaan baik seperti olahraga teratur dan istirahat cukup." }
    },
    stress: {
      high: { desc: "Skor menunjukkan kemungkinan besar mengalami tingkat stres yang tinggi yang dapat memengaruhi kesehatan fisik dan mental.", rec: "Sangat disarankan untuk mencari bantuan profesional dan menerapkan strategi manajemen stres." },
      medium: { desc: "Ada indikasi tingkat stres yang perlu diperhatikan.", rec: "Cobalah teknik manajemen stres seperti olahraga, meditasi, atau berbicara dengan orang terdekat." },
      low: { desc: "Tingkat stres Anda tampak terkendali.", rec: "Terus jaga keseimbangan antara pekerjaan, istirahat, dan aktivitas sosial." }
    }
  };
  return explanations[condition][risk];
};

export default function TesPage() {
  const [currentStep, setCurrentStep] = useState(-1); // -1 = Loading Config
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<any>(null);

  // Dynamic Questions based on MFO selection from Backend
  const [activeDemografi, setActiveDemografi] = useState<any[]>([]);
  const [stepsData, setStepsData] = useState<any[]>([]);
  const [formData, setFormData] = useState<Record<string, any>>({});

  useEffect(() => {
    // 1. Fetch konfigurasi fitur yang terpilih dari backend
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${apiUrl}/api/config`)
      .then(res => {
        if (!res.ok) throw new Error("API backend tidak berjalan");
        return res.json();
      })
      .then(data => {
        const selected = data.selected_features;
        
        // 2. Filter pertanyaan agar hanya menampilkan yang terpilih oleh model
        const filteredDemo = demografi.filter(q => selected.includes(q.id));
        const filteredTipi = tipi.filter(q => selected.includes(q.id));
        const filteredVcl = vcl.filter(q => selected.includes(q.id));
        
        setActiveDemografi(filteredDemo);
        
        // 3. Buat urutan halaman (steps) secara dinamis
        const generatedSteps = [
          { id: 'demografi', type: 'demografi' },
          ...filteredTipi.map((q, i) => ({ id: q.id, type: 'tipi', question: q, index: i + 1, total: filteredTipi.length })),
          ...filteredVcl.map((q, i) => ({ id: q.id, type: 'vcl', question: q, index: i + 1, total: filteredVcl.length }))
        ];
        
        setStepsData(generatedSteps);
        
        // 4. Inisialisasi state form dinamis
        setFormData({
          ...Object.fromEntries(filteredDemo.map(q => [q.id, q.defaultValue ?? q.options?.[0].val ?? 0])),
          ...Object.fromEntries(filteredTipi.map(q => [q.id, null])),
          ...Object.fromEntries(filteredVcl.map(q => [q.id, null]))
        });
        
        // Mulai tes
        setCurrentStep(0);
      })
      .catch(err => {
        console.error(err);
        alert("Gagal memuat konfigurasi tes dari server. Pastikan koneksi internet Anda stabil atau muat ulang halaman.");
      });
  }, []);

  const handleInputChange = (id: string, value: any) => {
    setFormData(prev => ({ ...prev, [id]: value }));
  };

  const nextStep = () => {
    if (currentStep < stepsData.length - 1) {
      setCurrentStep(prev => prev + 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      handleSubmit();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setCurrentStep(stepsData.length); // Pindah ke loading / result view
    
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) {
        throw new Error("Gagal menghubungi server API");
      }
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Terjadi kesalahan saat memproses jawaban Anda. Silakan coba kirim ulang.");
      setCurrentStep(0); // kembali ke awal jika gagal
    } finally {
      setIsSubmitting(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  // Variables for current view
  const isLoadingConfig = currentStep === -1;
  const isResultView = currentStep >= stepsData.length && !isLoadingConfig;
  const stepInfo = (!isResultView && !isLoadingConfig) ? stepsData[currentStep] : null;
  const progressPercentage = isLoadingConfig ? 0 : (isResultView ? 100 : ((currentStep) / stepsData.length) * 100);

  // Validation: User must select an option before proceeding on TIPI/VCL questions
  let isNextDisabled = false;
  if (stepInfo?.type === 'tipi' || stepInfo?.type === 'vcl') {
    if (formData[stepInfo.id] === null) {
      isNextDisabled = true;
    }
  }

  // Menunggu backend memberikan konfigurasi
  if (isLoadingConfig) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center">
        <div className="w-12 h-12 border-4 border-slate-200 border-t-[#6C63FF] rounded-full animate-spin mb-4"></div>
        <p className="text-slate-500 font-medium animate-pulse">Menghubungkan ke server prediksi...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans pb-32">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-20">
        <div className="max-w-4xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors">
            <ArrowLeft className="w-5 h-5" />
            <span className="font-medium hidden sm:inline">Kembali</span>
          </Link>
          <div className="flex items-center gap-2 text-[#6C63FF] font-bold">
            <Brain className="w-5 h-5" />
            <span className="hidden sm:inline">Tes Pulih.id</span>
          </div>
          <div className="w-16 sm:w-20 text-right text-sm font-medium text-slate-500">
            {!isResultView && `${Math.round(progressPercentage)}%`}
          </div>
        </div>
        
        {/* Progress Bar */}
        <div className="w-full bg-slate-100 h-1.5">
          <div 
            className="bg-[#6C63FF] h-1.5 transition-all duration-500 ease-out"
            style={{ width: `${progressPercentage}%` }}
          ></div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 pt-10">
        
        {/* VIEW: DEMOGRAFI */}
        {stepInfo?.type === 'demografi' && (
          <div className="animate-in fade-in slide-in-from-right-8 duration-500">
            <div className="mb-8">
              <span className="text-[#6C63FF] font-semibold text-sm tracking-wider uppercase mb-2 block">Bagian 1: Profil</span>
              <h1 className="text-3xl font-bold text-slate-900 mb-3">Informasi Demografis</h1>
              <p className="text-slate-600">Mohon lengkapi data demografis berikut. Data Anda akan diproses secara anonim.</p>
            </div>

            <div className="bg-white p-6 sm:p-8 rounded-3xl border border-slate-200 shadow-sm space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                {activeDemografi.map((field) => (
                  <div key={field.id} className="space-y-2">
                    <label className="text-sm font-semibold text-slate-700">{field.label}</label>
                    {field.type === "select" ? (
                      <select 
                        className="w-full p-3 rounded-xl border border-slate-300 bg-slate-50 focus:bg-white focus:ring-2 focus:ring-[#6C63FF]/50 focus:border-[#6C63FF] outline-none transition-all"
                        value={formData[field.id]}
                        onChange={(e) => handleInputChange(field.id, Number(e.target.value))}
                      >
                        {field.options?.map((opt: any) => (
                          <option key={opt.val} value={opt.val}>{opt.label}</option>
                        ))}
                      </select>
                    ) : (
                      <input 
                        type="number" 
                        min={field.min} 
                        max={field.max}
                        className="w-full p-3 rounded-xl border border-slate-300 bg-slate-50 focus:bg-white focus:ring-2 focus:ring-[#6C63FF]/50 focus:border-[#6C63FF] outline-none transition-all"
                        value={formData[field.id]}
                        onChange={(e) => handleInputChange(field.id, Number(e.target.value))}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* VIEW: TIPI (Per Pertanyaan) */}
        {stepInfo?.type === 'tipi' && (
          <div key={stepInfo.id} className="animate-in fade-in slide-in-from-right-8 duration-500">
            <div className="mb-8 text-center max-w-xl mx-auto">
              <span className="text-[#6C63FF] font-semibold text-sm tracking-wider uppercase mb-2 block">
                Bagian 2: Sifat Kepribadian ({stepInfo.index} dari {stepInfo.total})
              </span>
              <h2 className="text-2xl md:text-3xl font-bold text-slate-900 mb-3">{stepInfo.question.text}</h2>
              <p className="text-slate-500 inline-block bg-purple-50 px-4 py-1.5 rounded-full text-sm font-medium border border-purple-100">{stepInfo.question.desc}</p>
            </div>

            <div className="max-w-xl mx-auto space-y-3">
              {tipiOptions.map(opt => (
                <button
                  key={opt.val}
                  onClick={() => handleInputChange(stepInfo.id, opt.val)}
                  className={`w-full p-4 rounded-2xl border-2 text-left font-medium transition-all ${
                    formData[stepInfo.id] === opt.val 
                      ? "border-[#6C63FF] bg-[#6C63FF]/5 text-[#6C63FF] shadow-sm shadow-indigo-100" 
                      : "border-slate-200 bg-white text-slate-700 hover:border-purple-300 hover:bg-slate-50"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* VIEW: VCL (Per Pertanyaan) */}
        {stepInfo?.type === 'vcl' && (
          <div key={stepInfo.id} className="animate-in fade-in slide-in-from-right-8 duration-500">
            <div className="mb-8 text-center max-w-xl mx-auto">
              <span className="text-[#48CAE4] font-semibold text-sm tracking-wider uppercase mb-2 block">
                Bagian 3: Penggunaan Zat ({stepInfo.index} dari {stepInfo.total})
              </span>
              <h2 className="text-2xl md:text-3xl font-bold text-slate-900 mb-3">{stepInfo.question.text}</h2>
              <p className="text-slate-500 text-sm">Seberapa sering Anda menggunakan atau mengonsumsi ini?</p>
            </div>

            <div className="max-w-xl mx-auto space-y-3">
              {vclOptions.map(opt => (
                <button
                  key={opt.val}
                  onClick={() => handleInputChange(stepInfo.id, opt.val)}
                  className={`w-full p-4 rounded-2xl border-2 text-left font-medium transition-all ${
                    formData[stepInfo.id] === opt.val 
                      ? "border-[#48CAE4] bg-[#48CAE4]/5 text-[#009fb8] shadow-sm shadow-cyan-100" 
                      : "border-slate-200 bg-white text-slate-700 hover:border-cyan-300 hover:bg-slate-50"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Loading State during Submission */}
        {isResultView && isSubmitting && (
          <div className="flex flex-col items-center justify-center py-20 animate-in fade-in duration-500">
            <div className="w-16 h-16 border-4 border-slate-200 border-t-[#6C63FF] rounded-full animate-spin mb-6"></div>
            <h2 className="text-2xl font-bold text-slate-900 mb-2">Menganalisis Jawaban...</h2>
            <p className="text-slate-500">Menerapkan model Machine Learning pada data Anda.</p>
          </div>
        )}

        {/* VIEW: RESULT MOCKUP */}
        {isResultView && !isSubmitting && result && (
          <div className="animate-in fade-in zoom-in-95 duration-500">
            <div className="text-center mb-10">
              <div className="w-20 h-20 bg-green-100 text-green-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <CheckCircle className="w-10 h-10" />
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Analisis Selesai</h1>
              <p className="text-slate-600 max-w-lg mx-auto">
                Berikut adalah hasil prediksi risiko kesehatan mental Anda berdasarkan respons yang diberikan dan fitur yang dipilih oleh sistem.
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mb-10">
              {/* Depresi */}
              <div className={`p-6 rounded-3xl border flex flex-col ${result.depression.risk === 'high' ? 'bg-red-50 border-red-200' : result.depression.risk === 'medium' ? 'bg-orange-50 border-orange-200' : 'bg-green-50 border-green-200'}`}>
                <div className="text-3xl mb-4">😔</div>
                <h3 className="font-bold text-slate-900 mb-1">Depresi</h3>
                <div className="flex items-end gap-2 mb-4">
                  <span className="text-3xl font-extrabold">{(result.depression.prob * 100).toFixed(0)}%</span>
                  <span className="text-slate-500 mb-1 text-sm font-medium uppercase tracking-wider">{result.depression.risk}</span>
                </div>
                <div className="w-full bg-black/5 rounded-full h-2 mb-6">
                  <div className={`h-2 rounded-full ${result.depression.risk === 'high' ? 'bg-red-500' : result.depression.risk === 'medium' ? 'bg-orange-500' : 'bg-green-500'}`} style={{ width: `${result.depression.prob * 100}%` }}></div>
                </div>
                <div className="flex-grow">
                  <p className="text-sm text-slate-700 mb-3">{getExplanation('depression', result.depression.risk).desc}</p>
                  <div className="bg-white/60 p-3 rounded-xl border border-white/40">
                    <strong className="text-xs uppercase tracking-wider text-slate-500 block mb-1">Rekomendasi</strong>
                    <p className="text-sm font-medium text-slate-800">{getExplanation('depression', result.depression.risk).rec}</p>
                  </div>
                </div>
              </div>

              {/* Kecemasan */}
              <div className={`p-6 rounded-3xl border flex flex-col ${result.anxiety.risk === 'high' ? 'bg-red-50 border-red-200' : result.anxiety.risk === 'medium' ? 'bg-orange-50 border-orange-200' : 'bg-green-50 border-green-200'}`}>
                <div className="text-3xl mb-4">😰</div>
                <h3 className="font-bold text-slate-900 mb-1">Kecemasan</h3>
                <div className="flex items-end gap-2 mb-4">
                  <span className="text-3xl font-extrabold">{(result.anxiety.prob * 100).toFixed(0)}%</span>
                  <span className="text-slate-500 mb-1 text-sm font-medium uppercase tracking-wider">{result.anxiety.risk}</span>
                </div>
                <div className="w-full bg-black/5 rounded-full h-2 mb-6">
                  <div className={`h-2 rounded-full ${result.anxiety.risk === 'high' ? 'bg-red-500' : result.anxiety.risk === 'medium' ? 'bg-orange-500' : 'bg-green-500'}`} style={{ width: `${result.anxiety.prob * 100}%` }}></div>
                </div>
                <div className="flex-grow">
                  <p className="text-sm text-slate-700 mb-3">{getExplanation('anxiety', result.anxiety.risk).desc}</p>
                  <div className="bg-white/60 p-3 rounded-xl border border-white/40">
                    <strong className="text-xs uppercase tracking-wider text-slate-500 block mb-1">Rekomendasi</strong>
                    <p className="text-sm font-medium text-slate-800">{getExplanation('anxiety', result.anxiety.risk).rec}</p>
                  </div>
                </div>
              </div>

              {/* Stres */}
              <div className={`p-6 rounded-3xl border flex flex-col ${result.stress.risk === 'high' ? 'bg-red-50 border-red-200' : result.stress.risk === 'medium' ? 'bg-orange-50 border-orange-200' : 'bg-green-50 border-green-200'}`}>
                <div className="text-3xl mb-4">😣</div>
                <h3 className="font-bold text-slate-900 mb-1">Stres</h3>
                <div className="flex items-end gap-2 mb-4">
                  <span className="text-3xl font-extrabold">{(result.stress.prob * 100).toFixed(0)}%</span>
                  <span className="text-slate-500 mb-1 text-sm font-medium uppercase tracking-wider">{result.stress.risk}</span>
                </div>
                <div className="w-full bg-black/5 rounded-full h-2 mb-6">
                  <div className={`h-2 rounded-full ${result.stress.risk === 'high' ? 'bg-red-500' : result.stress.risk === 'medium' ? 'bg-orange-500' : 'bg-green-500'}`} style={{ width: `${result.stress.prob * 100}%` }}></div>
                </div>
                <div className="flex-grow">
                  <p className="text-sm text-slate-700 mb-3">{getExplanation('stress', result.stress.risk).desc}</p>
                  <div className="bg-white/60 p-3 rounded-xl border border-white/40">
                    <strong className="text-xs uppercase tracking-wider text-slate-500 block mb-1">Rekomendasi</strong>
                    <p className="text-sm font-medium text-slate-800">{getExplanation('stress', result.stress.risk).rec}</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-rose-50 border-2 border-rose-200 p-6 sm:p-8 rounded-3xl flex flex-col sm:flex-row gap-6 items-start text-rose-900">
              <div className="w-12 h-12 bg-rose-200 rounded-2xl flex items-center justify-center flex-shrink-0 text-rose-600">
                <AlertTriangle className="w-7 h-7" />
              </div>
              <div>
                <h4 className="text-lg font-bold mb-2">Peringatan & Disclaimer Medis</h4>
                <p className="text-sm text-rose-800/90 leading-relaxed mb-4">
                  Tes ini dikembangkan sebagai alat bantu informasi (<em>self-awareness</em>) berdasarkan model <em>Machine Learning</em>, dan <strong>bukan merupakan alat diagnosis klinis yang pasti</strong>. Hasil dari prediksi sistem ini <strong>tidak boleh digunakan untuk menggantikan peran psikolog, psikiater, maupun konselor profesional</strong>.
                </p>
                <p className="text-sm text-rose-800/90 leading-relaxed font-semibold">
                  Jika Anda merasakan tekanan emosional yang berat, dorongan untuk menyakiti diri sendiri, atau membutuhkan bantuan mendesak, mohon segera hubungi layanan gawat darurat (119) atau datangi layanan fasilitas kesehatan mental terdekat di kota Anda.
                </p>
              </div>
            </div>

            <div className="mt-10 text-center">
              <button 
                onClick={() => {
                  setCurrentStep(0);
                  setResult(null);
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
                className="bg-slate-900 text-white px-8 py-3 rounded-xl font-semibold hover:bg-slate-800 transition-colors"
              >
                Ulangi Tes
              </button>
            </div>
          </div>
        )}

      </main>

      {/* Floating Bottom Navigation */}
      {!isResultView && !isLoadingConfig && (
        <div className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-md border-t border-slate-200 p-4 z-20">
          <div className="max-w-3xl mx-auto flex justify-between items-center px-2">
            <button 
              onClick={prevStep}
              disabled={currentStep === 0}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-5 h-5" />
              <span className="hidden sm:inline">Sebelumnya</span>
            </button>
            
            <button 
              onClick={nextStep}
              disabled={isNextDisabled}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-white transition-all shadow-sm ${
                isNextDisabled 
                  ? 'bg-slate-300 cursor-not-allowed' 
                  : stepInfo?.type === 'vcl' ? 'bg-[#48CAE4] hover:bg-[#3bb5cd] shadow-cyan-500/30' : 'bg-[#6C63FF] hover:bg-[#5a52d5] shadow-indigo-500/30'
              }`}
            >
              {currentStep === stepsData.length - 1 ? "Kirim Prediksi" : "Selanjutnya"}
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
