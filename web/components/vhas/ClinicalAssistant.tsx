"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  CheckCircle2,
  HeartPulse,
  Loader2,
  Pill,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { useI18n, type Locale } from "@/components/vhas/i18n";

type Stage = "idle" | "processing" | "complete";

type CaseId = "dyspnea" | "afib" | "epigastric";

interface DemoCase {
  id: CaseId;
  label: string;
  patientId: string;
  complaint: string;
}

const demoCases: Record<Locale, DemoCase[]> = {
  en: [
    {
      id: "dyspnea",
      label: "66yo Male, Dyspnea (Fibrosis)",
      patientId: "ED-66712",
      complaint:
        "66-year-old male with progressive dyspnea and dry cough, known lung fibrosis.",
    },
    {
      id: "afib",
      label: "83yo Female, AFib w/ RVR",
      patientId: "ED-8392",
      complaint:
        "83-year-old female with a history of a permanent pacemaker. Presents with dizziness, weakness, and palpitations. Pacemaker interrogation notes Atrial Fibrillation.",
    },
    {
      id: "epigastric",
      label: "45yo Male, Epigastric Pain",
      patientId: "ED-5092",
      complaint:
        "45-year-old male presenting with sudden onset of severe, diffuse epigastric pain and nausea. History of heavy alcohol use.",
    },
  ],
  vi: [
    {
      id: "dyspnea",
      label: "Nam 66t, Khó thở (Xơ phổi)",
      patientId: "ED-66712",
      complaint:
        "Nam 66 tuổi khó thở tăng dần, ho khan, tiền sử xơ phổi.",
    },
    {
      id: "afib",
      label: "Nữ 83t, AFib RVR",
      patientId: "ED-8392",
      complaint:
        "Nữ 83 tuổi có máy tạo nhịp vĩnh viễn. Đến vì chóng mặt, yếu, hồi hộp. Kiểm tra máy tạo nhịp ghi nhận rung nhĩ.",
    },
    {
      id: "epigastric",
      label: "Nam 45t, Đau thượng vị",
      patientId: "ED-5092",
      complaint:
        "Nam 45 tuổi đau bụng trên dữ dội, buồn nôn. Tiền sử uống rượu nhiều.",
    },
  ],
};


export function ClinicalAssistant() {
  const { locale } = useI18n();
  const [patientId, setPatientId] = useState("");
  const [complaint, setComplaint] = useState("");
  const [stage, setStage] = useState<Stage>("idle");
  const [stepIndex, setStepIndex] = useState(0);
  const [selectedCaseId, setSelectedCaseId] = useState<CaseId>("dyspnea");
  const [hitlChoice, setHitlChoice] = useState<
    "pancreatitis" | "melena" | null
  >(null);
  const cases = demoCases[locale];
  const isVi = locale === "vi";

  const handleGenerate = () => {
    setStepIndex(0);
    setHitlChoice(null);
    setStage("processing");
  };

  const handleReset = useCallback(() => {
    setStage("idle");
    setStepIndex(0);
    setHitlChoice(null);
    setPatientId("");
    setComplaint("");
  }, []);

  const timelineSteps = useMemo(() => {
    if (selectedCaseId === "epigastric") {
      return [
        {
          title: isVi ? "🏥 Đánh giá Cấp cứu" : "🏥 Triage Assessment",
          loading: isVi ? "Đang đánh giá mức độ..." : "Evaluating acuity...",
          content: (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {isVi ? "Mức độ" : "Acuity Level"}
                </span>
                <Badge variant="critical" className="text-xs font-semibold">
                  {isVi ? "CAO" : "HIGH"}
                </Badge>
              </div>
              <div className="text-sm text-slate-700">
                <span className="font-semibold text-slate-900">
                  {isVi ? "Chẩn đoán nghi ngờ:" : "Suspected Condition:"}
                </span>{" "}
                {isVi
                  ? "Đau thượng vị nặng chưa rõ nguyên nhân."
                  : "Undifferentiated Severe Epigastric Pain."}
              </div>
              <p className="text-sm text-slate-600">
                {isVi
                  ? "Lý do: Đau bụng trên nặng và tiền sử rượu. Cần phân biệt viêm tụy cấp và xuất huyết loét dạ dày."
                  : "Rationale: Patient has severe upper abdominal pain with a history of alcohol use. Need to differentiate between Acute Pancreatitis and Severe Peptic Ulcer Bleeding."}
              </p>
            </div>
          ),
        },
        {
          title: isVi ? "🗂️ Sinh hiệu & EHR" : "🗂️ Vitals & EHR Retrieval",
          loading: isVi
            ? "Đang truy vấn tiền sử và sinh hiệu..."
            : "Querying patient history and vitals...",
          content: (
            <div className="space-y-3">
              <div className="grid gap-2 sm:grid-cols-3">
                {[
                  { label: "BP", value: "105/65" },
                  { label: "HR", value: "110" },
                  { label: "Temp", value: "38.2°C" },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                  >
                    <p className="text-xs font-semibold uppercase text-slate-400">
                      {item.label}
                    </p>
                    <p className="font-semibold text-slate-900">{item.value}</p>
                  </div>
                ))}
              </div>
              <p className="text-sm text-slate-600">
                {isVi
                  ? "Ghi chú EHR: Không có nội soi trước đó. Không ghi nhận dị ứng."
                  : "EHR Note: No prior endoscopies on file. No known allergies."}
              </p>
            </div>
          ),
        },
        {
          title: isVi ? "📋 Bàn giao Lâm sàng" : "📋 Final Handover Brief",
          loading: isVi ? "Đang tổng hợp tóm tắt..." : "Synthesizing clinical summary...",
          content: (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Badge variant="critical" className="text-xs font-semibold">
                  {isVi ? "🚨 NGUY CƠ CAO - EWS: 5" : "🚨 HIGH RISK - EWS: 5"}
                </Badge>
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  {isVi ? "Tổng kết" : "Clinical Summary"}
                </span>
              </div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {isVi ? "Độ tin cậy: 45%" : "Confidence Score: 45%"}
              </p>
              <p className="text-sm text-slate-700">
                {isVi
                  ? "Nam 45 tuổi đau thượng vị nặng, nhịp tim nhanh. Bác sĩ xác nhận đau lan ra sau lưng và nôn không kiểm soát. Nghi ngờ cao viêm tụy cấp (liên quan rượu)."
                  : "45yo male with severe epigastric pain and tachycardia. Clinician confirms pain radiating to the back with intractable vomiting. Highly suspicious for Acute Pancreatitis (Likely alcohol-induced)."}
              </p>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {isVi ? "Đề xuất tiếp theo" : "AI Proposed Next Steps"}
                </p>
                <p className="mt-2 text-sm text-slate-700">
                  {isVi
                    ? "Hội chẩn Nội/GI. NPO, bù dịch truyền tĩnh nhanh (Lactated Ringer&apos;s), chỉ định Lipase cấp cứu và siêu âm HSP."
                    : "Internal Medicine/GI Consult. NPO, start aggressive IV Fluid Resuscitation (Lactated Ringer&apos;s), order STAT Serum Lipase and RUQ Ultrasound."}
                </p>
              </div>
              <div className="flex items-center justify-end">
                <Button className="w-full sm:w-auto" onClick={handleReset}>
                  {isVi ? "Xác nhận & Tiếp tục" : "Acknowledge & Proceed"}
                </Button>
              </div>
            </div>
          ),
        },
      ];
    }

    if (selectedCaseId === "afib") {
      return [
        {
          title: isVi ? "🏥 Đánh giá Cấp cứu" : "🏥 Triage Assessment",
          loading: isVi ? "Đang đánh giá mức độ..." : "Evaluating acuity...",
          content: (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {isVi ? "Mức độ" : "Acuity Level"}
                </span>
                <Badge variant="critical" className="text-xs font-semibold">
                  {isVi ? "CAO" : "HIGH"}
                </Badge>
              </div>
              <div className="text-sm text-slate-700">
                <span className="font-semibold text-slate-900">
                  {isVi ? "Chẩn đoán nghi ngờ:" : "Suspected Condition:"}
                </span>{" "}
                {isVi
                  ? "Rung nhĩ có đáp thất nhanh (AFib RVR)."
                  : "Atrial Fibrillation with Rapid Ventricular Response (AFib w/ RVR)."}
              </div>
              <p className="text-sm text-slate-600">
                {isVi
                  ? "Lý do: Hồi hộp, yếu, tiền sử máy tạo nhịp."
                  : "Rationale: Based on palpitations, weakness, and pacemaker history."}
              </p>
            </div>
          ),
        },
        {
          title: isVi
            ? "🗂️ Sinh hiệu ban đầu & EHR"
            : "🗂️ Initial Vitals & EHR Retrieval",
          loading: isVi
            ? "Đang truy vấn tiền sử và sinh hiệu..."
            : "Querying patient history and vitals...",
          content: (
            <div className="space-y-3">
              <div className="grid gap-2 sm:grid-cols-3">
                {[
                  { label: "BP", value: "95/60" },
                  { label: "HR", value: "145" },
                  { label: "Temp", value: "36.8°C" },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                  >
                    <p className="text-xs font-semibold uppercase text-slate-400">
                      {item.label}
                    </p>
                    <p className="font-semibold text-slate-900">{item.value}</p>
                  </div>
                ))}
              </div>
              <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700">
                {isVi
                  ? "CẢNH BÁO: Huyết động không ổn định (HA thấp + Nhịp nhanh)."
                  : "CRITICAL ALERT: Patient is hemodynamically unstable (Hypotension + Severe Tachycardia)."}
              </div>
            </div>
          ),
        },
        {
          title: isVi ? "💊 Can thiệp Cấp cứu" : "💊 Emergency Dispensation",
          loading: isVi
            ? "Đang thực hiện can thiệp..."
            : "Dispensing emergency intervention...",
          content: (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm text-slate-700">
                <Pill className="h-4 w-4 text-blue-600" />
                <span className="font-semibold text-slate-900">Action Taken:</span>
                {isVi
                  ? "Tiêm tĩnh mạch 10mg Diltiazem."
                  : "Administered 10mg Diltiazem IV push."}
              </div>
              <p className="text-sm text-slate-600">
                {isVi
                  ? "Mục tiêu: Kiểm soát nhịp tim khẩn cấp."
                  : "Clinical Goal: Urgent heart rate control."}
              </p>
            </div>
          ),
        },
        {
          title: isVi
            ? "🗂️ Sinh hiệu sau can thiệp"
            : "🗂️ Post-Intervention Vitals Re-check",
          loading: isVi
            ? "Đang kiểm tra lại sinh hiệu..."
            : "Re-checking post-intervention vitals...",
          content: (
            <div className="space-y-3">
              <div className="grid gap-2 sm:grid-cols-2">
                {[
                  { label: "BP", value: "110/70" },
                  { label: "HR", value: "95" },
                ].map((item) => (
                  <div
                    key={item.label}
                    className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                  >
                    <p className="text-xs font-semibold uppercase text-slate-400">
                      {item.label}
                    </p>
                    <p className="font-semibold text-slate-900">{item.value}</p>
                  </div>
                ))}
              </div>
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-700">
                {isVi
                  ? "Huyết động cải thiện. Nhịp tim đã được kiểm soát sau Diltiazem."
                  : "Hemodynamics improved. Heart rate successfully controlled post-IV Diltiazem."}
              </div>
            </div>
          ),
        },
        {
          title: isVi ? "📋 Đối chiếu Thuốc" : "📋 Medication Reconciliation",
          loading: isVi
            ? "Đang đối chiếu tiền sử thuốc..."
            : "Reconciling medication history...",
          content: (
            <div className="space-y-3">
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-700">
                {isVi
                  ? "CẢNH BÁO: Không tuân thủ thuốc."
                  : "WARNING: Patient Non-Compliance Detected."}
              </div>
              <p className="text-sm text-amber-700">
                {isVi
                  ? "EHR và nhà thuốc cho thấy bệnh nhân ngừng Apixaban 3 ngày. Nguy cơ huyết khối cao."
                  : "EHR and pharmacy records indicate the patient has been out of their prescribed anticoagulant (Apixaban) for 3 days. High risk for thromboembolism."}
              </p>
            </div>
          ),
        },
        {
          title: isVi ? "📑 Bàn giao Lâm sàng" : "📑 Final Handover Brief",
          loading: isVi ? "Đang tổng hợp tóm tắt..." : "Synthesizing clinical summary...",
          content: (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <Badge variant="critical" className="text-xs font-semibold">
                  {isVi ? "🚨 NGUY CƠ CAO - EWS: 6" : "🚨 HIGH RISK - EWS: 6"}
                </Badge>
                <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                  {isVi ? "Tổng kết" : "Clinical Summary"}
                </span>
              </div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {isVi ? "Độ tin cậy: 92%" : "Confidence Score: 92%"}
              </p>
              <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700">
                {isVi ? "⚠️ CẢNH BÁO DỊ ỨNG: Penicillin" : "⚠️ ALLERGY WARNING: Penicillin"}
              </div>
              <p className="text-sm text-slate-700">
                {isVi
                  ? "Nữ 83 tuổi có máy tạo nhịp, AFib RVR và hạ huyết áp. Nhịp tim đã được kiểm soát sau Diltiazem. Đối chiếu thuốc: bỏ Apixaban 3 ngày. Bệnh nhân ổn định, sẵn sàng nhập viện và hội chẩn Tim mạch."
                  : "83yo female with pacemaker presented with AFib with RVR and hypotension. Heart rate now controlled post-IV Diltiazem. Med recon reveals 3-day non-compliance with Apixaban. Patient stabilized, ready for admission and Cardiology consult."}
              </p>
              <div className="grid gap-3 lg:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    {isVi ? "Tiền sử trích xuất" : "Extracted History"}
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-slate-700">
                    <li>
                      {isVi
                        ? "• Máy tạo nhịp vĩnh viễn, tiền sử AFib"
                        : "• Permanent pacemaker, AFib history"}
                    </li>
                    <li>
                      {isVi
                        ? "• Hạ huyết áp kèm nhịp nhanh"
                        : "• Hypotension with severe tachycardia"}
                    </li>
                    <li>
                      {isVi
                        ? "• Không tuân thủ thuốc chống đông"
                        : "• Recent anticoagulant non-compliance"}
                    </li>
                  </ul>
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    {isVi ? "Đề xuất tiếp theo" : "AI Proposed Next Steps"}
                  </p>
                  <ul className="mt-2 space-y-1 text-sm text-slate-700">
                    <li>
                      {isVi
                        ? "• Theo dõi telemetry và nhập viện"
                        : "• Initiate telemetry and admit for monitoring"}
                    </li>
                    <li>
                      {isVi
                        ? "• Hội chẩn Tim mạch và kế hoạch chống đông"
                        : "• Cardiology consult and anticoagulation plan"}
                    </li>
                    <li>
                      {isVi
                        ? "• Duy trì kiểm soát nhịp tim"
                        : "• Continue rate control as indicated"}
                    </li>
                  </ul>
                </div>
              </div>
              <div className="flex items-center justify-end">
                <Button className="w-full sm:w-auto" onClick={handleReset}>
                  {isVi ? "Xác nhận & Tiếp tục" : "Acknowledge & Proceed"}
                </Button>
              </div>
            </div>
          ),
        },
      ];
    }

    return [
      {
        title: isVi ? "🏥 Đánh giá Cấp cứu" : "🏥 Triage Assessment",
        loading: isVi ? "Đang đánh giá mức độ..." : "Evaluating acuity...",
        content: (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {isVi ? "Mức độ" : "Acuity Level"}
              </span>
              <Badge variant="critical" className="text-xs font-semibold">
                {isVi ? "CAO" : "HIGH"}
              </Badge>
            </div>
            <div className="text-sm text-slate-700">
              <span className="font-semibold text-slate-900">
                {isVi ? "Chẩn đoán nghi ngờ:" : "Suspected Condition:"}
              </span>{" "}
              {isVi ? "Đợt cấp xơ phổi." : "Lung Fibrosis Exacerbation"}
            </div>
            <p className="text-sm text-slate-600">
              {isVi
                ? "Lý do: Khó thở tăng dần và ho khan."
                : "Rationale: Based on progressive dyspnea and dry cough."}
            </p>
          </div>
        ),
      },
      {
        title: isVi ? "🗂️ Sinh hiệu & EHR" : "🗂️ Vitals & EHR Retrieval",
        loading: isVi
          ? "Đang truy vấn tiền sử và sinh hiệu..."
          : "Querying patient history and vitals...",
        content: (
          <div className="space-y-3">
            <div className="grid gap-2 sm:grid-cols-3">
              {[
                { label: "BP", value: "130/80" },
                { label: "HR", value: "105" },
                { label: "Temp", value: "37.2°C" },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                >
                  <p className="text-xs font-semibold uppercase text-slate-400">
                    {item.label}
                  </p>
                  <p className="font-semibold text-slate-900">{item.value}</p>
                </div>
              ))}
            </div>
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700">
              {isVi ? "CẢNH BÁO: SpO2 89% (Thiếu oxy)" : "CRITICAL ALERT: SpO2 89% (Hypoxia)"}
            </div>
            <p className="text-sm text-rose-600">
              {isVi
                ? "Bệnh nhân có dấu hiệu suy hô hấp nặng."
                : "Patient is showing signs of severe respiratory distress."}
            </p>
          </div>
        ),
      },
      {
        title: isVi ? "📋 Bàn giao Lâm sàng" : "📋 Final Handover Brief",
        loading: isVi ? "Đang tổng hợp tóm tắt..." : "Synthesizing clinical summary...",
        content: (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Badge variant="critical" className="text-xs font-semibold">
                {isVi ? "🚨 NGUY CƠ CAO - EWS: 6" : "🚨 HIGH RISK - EWS: 6"}
              </Badge>
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                {isVi ? "Tổng kết" : "Clinical Summary"}
              </span>
            </div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              {isVi ? "Độ tin cậy: 90%" : "Confidence Score: 90%"}
            </p>
            <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm font-semibold text-rose-700">
              {isVi ? "⚠️ CẢNH BÁO DỊ ỨNG: Penicillin" : "⚠️ ALLERGY WARNING: Penicillin"}
            </div>
            <p className="text-sm text-slate-700">
              {isVi
                ? "Nam 66 tuổi xơ phổi, khó thở nặng hơn. Ưu tiên cao. Sinh hiệu cho thấy thiếu oxy SpO2 89%. Sẵn sàng đánh giá và cân nhắc nhập viện."
                : "66-year-old male with lung fibrosis presents with worsening dyspnea. Triaged as high priority. Vitals notable for hypoxia with O2 saturation of 89%. Ready for physician evaluation for supplemental oxygen and consideration for admission."}
            </p>
            <div className="grid gap-3 lg:grid-cols-2">
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {isVi ? "Tiền sử trích xuất" : "Extracted History"}
                </p>
                <ul className="mt-2 space-y-1 text-sm text-slate-700">
                  <li>
                    {isVi
                      ? "• Khó thở tăng dần, ho khan"
                      : "• Progressive dyspnea with dry cough"}
                  </li>
                  <li>
                    {isVi
                      ? "• Xơ phổi, triệu chứng nặng hơn"
                      : "• Known lung fibrosis with recent decline"}
                  </li>
                  <li>{isVi ? "• Thiếu oxy (SpO2 89%)" : "• Hypoxia on arrival (SpO2 89%)"}</li>
                </ul>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {isVi ? "Đề xuất tiếp theo" : "AI Proposed Next Steps"}
                </p>
                <ul className="mt-2 space-y-1 text-sm text-slate-700">
                  <li>
                    {isVi
                      ? "• Cho oxy và theo dõi đáp ứng"
                      : "• Start supplemental oxygen and monitor response"}
                  </li>
                  <li>
                    {isVi
                      ? "• Cân nhắc nhập viện hỗ trợ hô hấp"
                      : "• Consider admission for respiratory support"}
                  </li>
                  <li>
                    {isVi ? "• Lặp lại sinh hiệu và ABG" : "• Repeat vitals and ABG as indicated"}
                  </li>
                </ul>
              </div>
            </div>
            <div className="flex items-center justify-end">
              <Button className="w-full sm:w-auto" onClick={handleReset}>
                {isVi ? "Xác nhận & Tiếp tục" : "Acknowledge & Proceed"}
              </Button>
            </div>
          </div>
        ),
      },
    ];
  }, [handleReset, isVi, selectedCaseId]);

  const stepDurations = useMemo(
    () => timelineSteps.map(() => 1500),
    [timelineSteps]
  );

  const isHitlCase = selectedCaseId === "epigastric";
  const isHitlResolved = hitlChoice === "pancreatitis";

  useEffect(() => {
    if (stage !== "processing") return;
    if (isHitlCase && stepIndex === 1 && !isHitlResolved) return;

    const timer = setTimeout(() => {
      if (stepIndex >= timelineSteps.length - 1) {
        setStage("complete");
        return;
      }
      setStepIndex((prev) => Math.min(prev + 1, timelineSteps.length - 1));
    }, stepDurations[stepIndex] ?? 1500);

    return () => clearTimeout(timer);
  }, [
    stage,
    stepIndex,
    stepDurations,
    timelineSteps.length,
    isHitlCase,
    isHitlResolved,
  ]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">
            {isVi ? "Trợ lý Lâm sàng" : "Clinical Assistant"}
          </h2>
          <p className="text-sm text-slate-500">
            {isVi
              ? "Tiếp nhận và tạo bàn giao lâm sàng sẵn sàng phục vụ."
              : "Intake triage context and generate a ready-to-serve handover brief."}
          </p>
        </div>
        <Badge variant="info" className="flex items-center gap-2">
          <HeartPulse className="h-4 w-4" />
          {isVi ? "Nội khoa - ED" : "Internal Medicine - ED"}
        </Badge>
      </div>

      <AnimatePresence mode="wait">
        {stage === "idle" && (
          <motion.div
            key="intake"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>{isVi ? "Form Tiếp nhận ED" : "ED Intake Form"}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4 md:grid-cols-[180px_1fr]">
                  <div className="space-y-2">
                    <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      {isVi ? "Mã bệnh nhân" : "Patient ID"}
                    </label>
                    <Input
                      value={patientId}
                      onChange={(event) => setPatientId(event.target.value)}
                      placeholder={isVi ? "ED-XXXX" : "ED-XXXX"}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      {isVi ? "Lý do vào viện & Bệnh sử" : "Chief Complaint & Brief History"}
                    </label>
                    <Textarea
                      value={complaint}
                      onChange={(event) => setComplaint(event.target.value)}
                      placeholder={
                        isVi
                          ? "Nhập tóm tắt tiếp nhận ngắn gọn..."
                          : "Enter concise ED intake summary..."
                      }
                    />
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {cases.map((item) => (
                    <button
                      key={item.id}
                      className={cn(
                        "rounded-full border px-3 py-1 text-xs font-semibold transition",
                        selectedCaseId === item.id
                          ? "border-blue-200 bg-blue-50 text-blue-700"
                          : "border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:text-blue-700"
                      )}
                      onClick={() => {
                        setPatientId(item.patientId);
                        setComplaint(item.complaint);
                        setSelectedCaseId(item.id);
                        setHitlChoice(null);
                      }}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-xs text-slate-500">
                    {isVi
                      ? "Bàn giao chỉ hỗ trợ quyết định và không thay thế bác sĩ."
                      : "Generated briefs support clinical decisions and do not replace physician judgment."}
                  </p>
                  <Button size="lg" onClick={handleGenerate}>
                    {isVi ? "Tạo Tóm tắt Lâm sàng" : "Generate Pre-Clinical Brief"}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence mode="wait">
        {(stage === "processing" || stage === "complete") && (
          <motion.div
            key="timeline"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.3 }}
            className="relative"
          >
            <div className="absolute left-4 top-4 h-full w-px bg-slate-200" />
            <div className="space-y-6">
              {timelineSteps.map((step, index) => {
                if (index > stepIndex) return null;
                const isActive = index === stepIndex && stage === "processing";
                const isComplete = index < stepIndex || stage === "complete";

                return (
                  <motion.div
                    key={step.title}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    whileHover={{ y: -4 }}
                    className="relative pl-12"
                  >
                    <div
                      className={cn(
                        "absolute left-2.5 top-6 flex h-6 w-6 items-center justify-center rounded-full border transition-shadow",
                        isComplete
                          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                          : "border-blue-200 bg-blue-50 text-blue-700",
                        isActive && "shadow-[0_0_0_6px_rgba(59,130,246,0.15)]"
                      )}
                    >
                      {isComplete ? (
                        <CheckCircle2 className="h-4 w-4" />
                      ) : (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      )}
                    </div>
                    <Card
                      className={cn(
                        "border-slate-200 transition-all",
                        isActive && "ring-1 ring-blue-100 shadow-md",
                        isComplete && "hover:shadow-lg"
                      )}
                    >
                      <CardHeader className="space-y-1">
                        <CardTitle className="text-base font-semibold text-slate-900">
                          {step.title}
                        </CardTitle>
                        <p className="text-xs text-slate-500">
                          {isComplete
                            ? isVi
                              ? "Đã hoàn thành"
                              : "Completed"
                            : isVi
                            ? "Đang xử lý"
                            : "Processing"}
                        </p>
                      </CardHeader>
                      <CardContent>
                        {isComplete ? (
                          step.content
                        ) : (
                          <div className="flex items-center gap-2 text-sm text-slate-500">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            {step.loading}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </motion.div>
                );
              })}
            </div>
            {isHitlCase && stepIndex >= 1 && stage === "processing" && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="relative pl-12"
              >
                <div
                  className={cn(
                    "absolute left-2.5 top-6 flex h-6 w-6 items-center justify-center rounded-full border border-amber-200 bg-amber-50 text-amber-700",
                    isHitlResolved && "opacity-60"
                  )}
                >
                  <span className="text-xs">!</span>
                </div>
                <Card
                  className={cn(
                    "border-amber-200 bg-amber-50/60",
                    isHitlResolved && "opacity-70"
                  )}
                >
                  <CardHeader className="space-y-1">
                    <CardTitle className="text-base font-semibold text-amber-800">
                      {isVi ? "⚠️ Cảnh báo Bất định VHAS" : "⚠️ VHAS Uncertainty Alert"}
                    </CardTitle>
                    <p className="text-xs text-amber-700">
                      {isVi ? "Độ tin cậy: 45%" : "Confidence Score: 45%"}
                    </p>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm text-amber-800">
                      {isVi
                        ? "Thiếu thông tin quan trọng để chọn đúng đường nội khoa. Vui lòng xác nhận dấu hiệu chính:"
                        : "Critical clinical data is missing to determine the correct Internal Medicine pathway. Please confirm the primary physical/history finding:"}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        disabled={isHitlResolved}
                        onClick={() => setHitlChoice("pancreatitis")}
                      >
                        {isVi
                          ? "Đau lan ra sau lưng + Nôn không kiểm soát"
                          : "Pain radiates to back + Intractable vomiting"}
                      </Button>
                      <Button
                        variant="outline"
                        disabled={isHitlResolved}
                        onClick={() => setHitlChoice("melena")}
                      >
                        {isVi
                          ? "Tiền sử phân đen (Melena)"
                          : "History of black tarry stools (Melena)"}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
            {stage === "complete" && (
              <div className="mt-6 flex justify-end">
                <Button variant="outline" onClick={handleReset}>
                  {isVi ? "Bắt đầu ca mới" : "Start New Intake"}
                </Button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
