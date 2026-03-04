"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Sidebar, type TabKey } from "@/components/vhas/Sidebar";
import { ClinicalAssistant } from "@/components/vhas/ClinicalAssistant";
import { TraceExplorer } from "@/components/vhas/TraceExplorer";
import { SystemDashboard } from "@/components/vhas/SystemDashboard";
import { I18nProvider, LanguageToggle, useI18n } from "@/components/vhas/i18n";
function HomeContent() {
	const [activeTab, setActiveTab] = useState<TabKey>("assistant");
	const { locale } = useI18n();

	return (
		<div className="min-h-screen bg-slate-100 text-slate-900">
			<div className="relative isolate flex min-h-screen">
				<div className="pointer-events-none absolute inset-0 -z-10">
					<div className="absolute left-10 top-10 h-72 w-72 rounded-full bg-blue-200/40 blur-3xl" />
					<div className="absolute bottom-20 right-10 h-72 w-72 rounded-full bg-emerald-200/30 blur-3xl" />
					<div className="absolute inset-x-0 top-0 h-48 bg-gradient-to-b from-white/80 to-transparent" />
				</div>

				<div className="hidden w-[280px] shrink-0 lg:block">
					<Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
				</div>

				<main className="flex min-h-screen flex-1 flex-col">
					<header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white/80 px-6 py-4 backdrop-blur">
						<div>
							<p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
								{locale === "vi"
									? "Hỗ trợ nhận thức khoa cấp cứu"
									: "Emergency Department Cognitive Support"}
							</p>
							<h1 className="text-2xl font-semibold text-slate-900">
								{locale === "vi"
									? "Bảng điều hành lâm sàng VHAS"
									: "VHAS Clinical Operations Console"}
							</h1>
						</div>
						<div className="flex flex-wrap items-center gap-2">
							<Badge variant="info">
								{locale === "vi" ? "Tim / Hô hấp / Tiêu hóa" : "Cardio / Respiratory / GI"}
							</Badge>
							<Badge variant="success">{locale === "vi" ? "Đang thí điểm" : "Live Pilot"}</Badge>
							<Badge variant="warning">
								{locale === "vi" ? "Sẵn sàng kiểm toán" : "Audit Ready"}
							</Badge>
							<LanguageToggle />
						</div>
					</header>

					<div className="flex-1 px-6 py-6 lg:px-10">
						<div className="mb-6 flex gap-3 lg:hidden">
							<Sidebar
								activeTab={activeTab}
								onTabChange={setActiveTab}
								variant="compact"
							/>
						</div>

						{activeTab === "assistant" && <ClinicalAssistant />}
						{activeTab === "trace" && <TraceExplorer />}
						{activeTab === "dashboard" && <SystemDashboard />}
					</div>
				</main>
			</div>
		</div>
	);
}

export default function Home() {
	return (
		<I18nProvider>
			<HomeContent />
		</I18nProvider>
	);
}
