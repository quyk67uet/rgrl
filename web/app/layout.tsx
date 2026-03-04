import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
	subsets: ["latin"],
	variable: "--font-sans",
});

export const metadata: Metadata = {
	title: "VHAS | Vietnam Health-Agent System",
	description:
		"Clinical decision support dashboard for emergency medicine workflows.",
};

export default function RootLayout({
	children,
}: Readonly<{ children: React.ReactNode }>) {
	return (
		<html lang="en" translate="no">
			<body className={`${inter.variable} antialiased`} translate="no">
				{children}
			</body>
		</html>
	);
}


