"use client";

import { motion } from "framer-motion";
import { BarChart3, CircleDollarSign, Gauge, ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useI18n } from "@/components/vhas/i18n";

const distribution = [
  { label: "Cardio", value: 42, color: "#1d4ed8" },
  { label: "Respiratory", value: 33, color: "#0ea5e9" },
  { label: "GI", value: 25, color: "#10b981" },
];

export function SystemDashboard() {
  const { locale } = useI18n();
  const isVi = locale === "vi";
  const kpis = [
    {
      label: isVi ? "Thời gian quy trình TB" : "Avg. Workflow Duration",
      value: isVi ? "50.5 phút" : "50.5 mins",
      delta: isVi ? "Giảm 47%" : "47% lower",
      tone: "success" as const,
      icon: Gauge,
    },
    {
      label: isVi ? "Tỷ lệ thành công" : "Success Rate",
      value: "87.5%",
      delta: isVi ? "Ổn định" : "Stable",
      tone: "info" as const,
      icon: ShieldCheck,
    },
    {
      label: isVi ? "Tỷ lệ trùng lặp" : "Redundancy Rate",
      value: "0.8%",
      delta: isVi ? "Thấp" : "Low",
      tone: "success" as const,
      icon: BarChart3,
    },
    {
      label: isVi ? "Chi phí TB" : "Avg. Resource Cost",
      value: "$105",
      delta: isVi ? "Giảm 43%" : "43% lower",
      tone: "warning" as const,
      icon: CircleDollarSign,
    },
  ];

  const adaptivity = [
    { label: isVi ? "Quy trình cố định" : "Static Workflow", value: 6 },
    { label: isVi ? "VHAS (Đơn giản)" : "VHAS (Simple)", value: 3 },
    { label: isVi ? "VHAS (Phức tạp)" : "VHAS (Complex)", value: 5 },
  ];
  const donutStyle = {
    background: `conic-gradient(${distribution
      .map((item, index) => {
        const start = distribution
          .slice(0, index)
          .reduce((acc, cur) => acc + cur.value, 0);
        const end = start + item.value;
        return `${item.color} ${start}% ${end}%`;
      })
      .join(", ")})`,
  };

  const distributionLabel = (label: string) => {
    if (!isVi) return label;
    if (label === "Cardio") return "Tim mạch";
    if (label === "Respiratory") return "Hô hấp";
    if (label === "GI") return "Tiêu hóa";
    return label;
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">
          {isVi ? "Bảng Hệ thống" : "System Dashboard"}
        </h2>
        <p className="text-sm text-slate-500">
          {isVi
            ? "Kết quả vận hành và chỉ số giá trị của VHAS."
            : "Operational outcomes and value metrics for VHAS deployments."}
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {kpis.map((kpi) => {
          const Icon = kpi.icon;
          return (
            <motion.div
              key={kpi.label}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Card>
                <CardContent className="space-y-3 pt-6">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                      {kpi.label}
                    </p>
                    <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100">
                      <Icon className="h-5 w-5 text-slate-600" />
                    </span>
                  </div>
                  <div className="text-2xl font-semibold text-slate-900">{kpi.value}</div>
                  <Badge variant={kpi.tone}>{kpi.delta}</Badge>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>{isVi ? "Phân tích thích ứng" : "Adaptivity Analysis"}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              {adaptivity.map((item) => (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>{item.label}</span>
                    <span>{item.value} {isVi ? "bước" : "steps"}</span>
                  </div>
                  <div className="h-3 rounded-full bg-slate-100">
                    <div
                      className="h-3 rounded-full bg-blue-600"
                      style={{ width: `${(item.value / 6) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              {isVi
                ? "VHAS rút ngắn quy trình cho ca đơn giản nhưng vẫn bảo toàn đường leo thang."
                : "VHAS reduces workflow length for low-complexity cases while preserving escalation paths."}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{isVi ? "Phân bố ca bệnh" : "Case Distribution"}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-6 md:flex-row md:items-center">
            <div className="relative h-40 w-40 rounded-full" style={donutStyle}>
              <div className="absolute inset-5 rounded-full bg-white" />
            </div>
            <div className="space-y-3">
              {distribution.map((item) => (
                <div key={item.label} className="flex items-center gap-3 text-sm">
                  <span
                    className="h-3 w-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <div className="flex-1 text-slate-700">
                    {distributionLabel(item.label)}
                  </div>
                  <div className="text-slate-500">{item.value}%</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
