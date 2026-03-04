"use client";

import { Stethoscope, FileSearch, Gauge } from "lucide-react";

import { cn } from "@/lib/utils";
import { useI18n } from "@/components/vhas/i18n";

export type TabKey = "assistant" | "trace" | "dashboard";

const navItems = {
  en: [
    {
      key: "assistant" as const,
      label: "Clinical Assistant",
      description: "ED intake and handover",
      icon: Stethoscope,
    },
    {
      key: "trace" as const,
      label: "Trace Explorer",
      description: "Explainability audit",
      icon: FileSearch,
    },
    {
      key: "dashboard" as const,
      label: "System Dashboard",
      description: "Operational metrics",
      icon: Gauge,
    },
  ],
  vi: [
    {
      key: "assistant" as const,
      label: "Trợ lý Lâm sàng",
      description: "Tiếp nhận ED và bàn giao",
      icon: Stethoscope,
    },
    {
      key: "trace" as const,
      label: "Truy vết Quyết định",
      description: "Kiểm toán giải thích",
      icon: FileSearch,
    },
    {
      key: "dashboard" as const,
      label: "Bảng Hệ thống",
      description: "Số liệu vận hành",
      icon: Gauge,
    },
  ],
};

interface SidebarProps {
  activeTab: TabKey;
  onTabChange: (tab: TabKey) => void;
  variant?: "full" | "compact";
}

export function Sidebar({
  activeTab,
  onTabChange,
  variant = "full",
}: SidebarProps) {
  const { locale } = useI18n();
  const items = navItems[locale];

  if (variant === "compact") {
    return (
      <div className="flex flex-wrap gap-2 rounded-xl border border-slate-200 bg-white/90 p-3">
        {items.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onTabChange(item.key)}
              className={cn(
                "flex items-center gap-2 rounded-full px-3 py-2 text-xs font-semibold transition",
                isActive
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <aside className="flex h-full flex-col justify-between gap-6 border-r border-slate-200 bg-white/90 px-5 py-6">
      <div className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            VHAS
          </div>
          <div className="text-lg font-semibold text-slate-900">
            {locale === "vi"
              ? "Hệ thống Tác tử Y tế Việt Nam"
              : "Vietnam Health-Agent System"}
          </div>
          <p className="text-sm text-slate-500">
            {locale === "vi"
              ? "Hỗ trợ nhận thức cho quy trình cấp cứu nội khoa."
              : "Cognitive support for ED internal medicine workflows."}
          </p>
        </div>
        <nav className="space-y-2">
          {items.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.key;
            return (
              <button
                key={item.key}
                onClick={() => onTabChange(item.key)}
                className={cn(
                  "flex w-full items-center gap-3 rounded-xl border px-3 py-3 text-left transition",
                  isActive
                    ? "border-blue-200 bg-blue-50 text-blue-900"
                    : "border-transparent hover:border-slate-200 hover:bg-slate-50"
                )}
              >
                <span
                  className={cn(
                    "flex h-10 w-10 items-center justify-center rounded-lg",
                    isActive ? "bg-blue-100" : "bg-slate-100"
                  )}
                >
                  <Icon className="h-5 w-5" />
                </span>
                <span className="space-y-1">
                  <span className="block text-sm font-semibold">{item.label}</span>
                  <span className="block text-xs text-slate-500">
                    {item.description}
                  </span>
                </span>
              </button>
            );
          })}
        </nav>
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600">
        {locale === "vi"
          ? "Thiết kế sẵn sàng phục vụ giúp thông tin lâm sàng rõ ràng, ngắn gọn, dễ hành động."
          : "Ready-to-serve design keeps clinical context clear, concise, and actionable."}
      </div>
    </aside>
  );
}
